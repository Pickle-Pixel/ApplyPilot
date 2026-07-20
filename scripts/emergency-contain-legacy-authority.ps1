[CmdletBinding(DefaultParameterSetName = 'Inspect')]
param(
  [Parameter(ParameterSetName = 'Inspect')][switch]$Inspect,
  [Parameter(Mandatory, ParameterSetName = 'Contain')][switch]$Contain,
  [Parameter(Mandatory, ParameterSetName = 'Probe')][string]$CommandLineProbe,
  [Parameter(Mandatory, ParameterSetName = 'Evaluate')][string]$EvaluateAfterStateJson,
  [Parameter(Mandatory, ParameterSetName = 'DefinitionImport')][switch]$DefinitionImport,
  [string]$AdapterPath,
  [string[]]$WrapperRoot,
  [string]$ConsoleCommand
)

$ErrorActionPreference = 'Stop'
$script:KnownAcquisitionTaskNamePattern = '(?i)^(?:ApplyPilot ApplyCycle|ApplyPilotKeepAlive|ApplyPilotFleet-FleetAgent)$'
$script:KnownAcquisitionServiceNamePattern = '(?i)^ApplyPilotWorkday$'
$script:AcquisitionConsoleBasenames = @(
  'applypilot-fleet-apply', 'applypilot-fleet-apply-home',
  'applypilot-fleet-linkedin', 'applypilot-fleet-linkedin-home',
  'applypilot-workday-onboard', 'applypilot-workday-rollout'
)
$script:AcquisitionPythonModules = @(
  'applypilot.apply.launcher',
  'applypilot.fleet.apply_home_main', 'applypilot.fleet.apply_worker_main',
  'applypilot.fleet.linkedin_home_main', 'applypilot.fleet.linkedin_worker_main',
  'applypilot.fleet.workday_onboard_main', 'applypilot.fleet.workday_rollout_main'
)
$script:AcquisitionPythonScriptBasenames = @(
  'apply_home_main.py', 'apply_worker_main.py',
  'linkedin_home_main.py', 'linkedin_worker_main.py',
  'workday_onboard_main.py', 'workday_rollout_main.py'
)
$script:AcquisitionWrapperBasenames = @(
  'run-fleet-worker.ps1', 'run-fleet-worker.cmd', 'run-fleet-workers.ps1',
  'run-tarpon-linkedin.ps1',
  'keepalive-apply.ps1', 'load-canary-home.ps1', 'load-canary-remote.ps1'
)
$script:GeneratedAcquisitionWrapperBasenames = @(
  'apply-cycle-task.ps1', 'fleet-agent-task.ps1',
  'apply-worker-m2.cmd', 'linkedin-m2.bat'
)
$script:SensitiveAssignment = '(?im)^\s*(?:\$env:(?:FLEET_PG_DSN|APPLYPILOT_FLEET_DSN|DATABASE_URL|APPLYPILOT_OPENAI_API_KEY|ANTHROPIC_API_KEY)\s*=.*|set\s+"?(?:FLEET_PG_DSN|APPLYPILOT_FLEET_DSN|DATABASE_URL|APPLYPILOT_OPENAI_API_KEY|ANTHROPIC_API_KEY)\s*=.*)$'
$script:SensitiveUri = '(?i)(?:postgres(?:ql)?://|host\s*=)[^\r\n''"]+'
$script:CredentialReferences = @(
  'FLEET_PG_DSN', 'APPLYPILOT_FLEET_DSN', 'DATABASE_URL',
  'APPLYPILOT_OPENAI_API_KEY', 'ANTHROPIC_API_KEY'
)
$script:SensitiveIdentifier = '(?i)(?<![a-z0-9_])(?:' +
  (($script:CredentialReferences | ForEach-Object { [regex]::Escape($_) }) -join '|') +
  ')(?![a-z0-9_])'
$script:Failures = [Collections.Generic.List[object]]::new()
$script:SkippedActions = [Collections.Generic.List[object]]::new()
$script:DefinitionImportMode = $PSCmdlet.ParameterSetName -eq 'DefinitionImport'
$script:WrapperBeforeLeafOpenHook = $null
$script:WrapperAfterComponentOpenHook = $null
$script:ActiveWrapperMutationLease = $null

$adapterInjected = $PSBoundParameters.ContainsKey('AdapterPath')
$consoleInjected = $PSBoundParameters.ContainsKey('ConsoleCommand')
$wrapperRootInjected = $PSBoundParameters.ContainsKey('WrapperRoot') -and
  $PSCmdlet.ParameterSetName -in @('Inspect', 'Contain')
if ($adapterInjected -or $consoleInjected -or $wrapperRootInjected) {
  $reasons = @()
  if ($adapterInjected) { $reasons += 'adapter_injected' }
  if ($consoleInjected) { $reasons += 'console_command_injected' }
  if ($wrapperRootInjected) { $reasons += 'wrapper_root_injected' }
  [ordered]@{
    schema_version = 3
    mode = 'test'
    operation = if ($Contain) { 'contain' } else { 'inspect' }
    operational = $false
    non_operational_reasons = $reasons
    success = $false
    rejection = 'injected_execution_seam_disabled'
    evidence_deleted = $false
  } | ConvertTo-Json -Compress
  exit 2
}

function Get-TextDigest([string]$Text) {
  if ([string]::IsNullOrEmpty($Text)) { return $null }
  $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
  $hash = [Security.Cryptography.SHA256]::HashData($bytes)
  return [Convert]::ToHexString($hash).ToLowerInvariant()
}

function Get-FileDigest([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-ProcessStableIdentityDigest($Process) {
  return Get-TextDigest ("{0}|{1}|{2}|{3}" -f
    [string]$Process.ProcessId,
    [string]$Process.CreationDate,
    [string]$Process.ExecutablePath,
    [string]$Process.CommandLine)
}

function ConvertFrom-DeterministicCommandLine([string]$CommandLine) {
  $tokens = [Collections.Generic.List[string]]::new()
  $current = [Text.StringBuilder]::new()
  $inQuotes = $false
  $tokenStarted = $false
  for ($i = 0; $i -lt $CommandLine.Length; $i++) {
    $character = $CommandLine[$i]
    if ($character -eq '\') {
      $slashCount = 0
      while ($i -lt $CommandLine.Length -and $CommandLine[$i] -eq '\') {
        $slashCount++
        $i++
      }
      if ($i -lt $CommandLine.Length -and $CommandLine[$i] -eq '"') {
        for ($j = 0; $j -lt [Math]::Floor($slashCount / 2); $j++) {
          [void]$current.Append('\')
        }
        if ($slashCount % 2 -eq 0) { $inQuotes = -not $inQuotes } else { [void]$current.Append('"') }
        $tokenStarted = $true
      } else {
        for ($j = 0; $j -lt $slashCount; $j++) { [void]$current.Append('\') }
        $i--
      }
      continue
    }
    if ($character -eq '"') {
      $inQuotes = -not $inQuotes
      $tokenStarted = $true
      continue
    }
    if ([char]::IsWhiteSpace($character) -and -not $inQuotes) {
      if ($tokenStarted) {
        $tokens.Add($current.ToString())
        [void]$current.Clear()
        $tokenStarted = $false
      }
      continue
    }
    [void]$current.Append($character)
    $tokenStarted = $true
  }
  if ($inQuotes) { throw 'unterminated command-line quote' }
  if ($tokenStarted) { $tokens.Add($current.ToString()) }
  return @($tokens)
}

function Get-CommandTokenBasename([string]$Token) {
  return [IO.Path]::GetFileName(($Token -replace '/', '\')).ToLowerInvariant()
}

function Get-CommandTokenStem([string]$Token) {
  $basename = Get-CommandTokenBasename $Token
  if ($basename.EndsWith('.exe')) { return $basename.Substring(0, $basename.Length - 4) }
  return $basename
}

function Test-IsPythonLauncherStem([string]$Stem) {
  return $Stem -match '^(?:pyw?|pythonw?(?:\d(?:\.\d+)?t?)?)$'
}

function Test-ExecutableTokenHasEnvironmentSelectedBasename([string]$Token) {
  $basename = Get-CommandTokenBasename $Token
  return $basename -match '(?i)(?:%|!|\$(?:\{)?env:)'
}

function Test-IsAcquisitionWrapperToken([string]$Token) {
  $basename = Get-CommandTokenBasename $Token
  return $script:AcquisitionWrapperBasenames -contains $basename -or
    $script:GeneratedAcquisitionWrapperBasenames -contains $basename
}

function Test-IsAcquisitionPythonScript([string]$Token) {
  $basename = Get-CommandTokenBasename $Token
  if ($script:AcquisitionPythonScriptBasenames -contains $basename) { return $true }
  if ($basename -ne 'launcher.py') { return $false }
  return ($Token -replace '/', '\') -match '(?i)(?:^|\\)apply\\launcher\.py$'
}

function Test-TextContainsBoundedAcquisitionIndicator([string]$Text) {
  $indicators = @(
    $script:AcquisitionWrapperBasenames +
    $script:GeneratedAcquisitionWrapperBasenames +
    $script:AcquisitionPythonScriptBasenames +
    $script:AcquisitionPythonModules +
    @('applypilot', 'applypilot.cli', 'apply\launcher.py')
  )
  foreach ($indicator in $indicators) {
    $escaped = [regex]::Escape($indicator)
    if ($Text -match "(?i)(?<![a-z0-9_.-])$escaped(?![a-z0-9_.-])") { return $true }
  }
  return $false
}

function Test-TokensContainAcquisitionIndicator([string[]]$Tokens) {
  foreach ($token in @($Tokens)) {
    if (Test-TextContainsBoundedAcquisitionIndicator $token) { return $true }
  }
  return $false
}

function Test-TokensContainExactAcquisitionIndicator([string[]]$Tokens) {
  $indicators = @(
    $script:AcquisitionConsoleBasenames +
    @($script:AcquisitionConsoleBasenames | ForEach-Object { "$_.exe" }) +
    $script:AcquisitionPythonModules +
    $script:AcquisitionPythonScriptBasenames +
    $script:AcquisitionWrapperBasenames +
    $script:GeneratedAcquisitionWrapperBasenames
  )
  foreach ($token in @($Tokens)) {
    foreach ($indicator in $indicators) {
      $escaped = [regex]::Escape($indicator)
      if ($token -match "(?i)(?<![a-z0-9_.-])$escaped(?![a-z0-9_.-])") { return $true }
    }
  }
  return $false
}

function Test-CommandTextContainsAcquisitionIndicator([string]$CommandLine) {
  return Test-TextContainsBoundedAcquisitionIndicator $CommandLine
}

function Get-AmbiguousOrBenignClassification([string[]]$Tokens) {
  if (Test-TokensContainAcquisitionIndicator $Tokens) { return 'ambiguous' }
  return 'benign'
}

function Get-EmbeddedCommandTokens([string[]]$PayloadTokens) {
  $payload = @($PayloadTokens)
  if ($payload.Count -eq 0) { return @() }
  if ($payload[0] -notmatch '\s') { return $payload }
  return @(ConvertFrom-DeterministicCommandLine $payload[0])
}

function Get-WrapperCommandClassification(
  [string[]]$PayloadTokens,
  [string[]]$AllowedInvocationPrefixes = @()
) {
  try {
    $commandTokens = @(Get-EmbeddedCommandTokens $PayloadTokens)
  } catch {
    return Get-AmbiguousOrBenignClassification $PayloadTokens
  }
  if ($commandTokens.Count -eq 0) { return 'benign' }

  $index = 0
  if ($AllowedInvocationPrefixes -contains $commandTokens[0].ToLowerInvariant()) { $index++ }
  $hasControlSyntax = @($commandTokens | Where-Object { $_ -match '[;|<>\r\n]' }).Count -gt 0
  if (-not $hasControlSyntax -and
      $index -lt $commandTokens.Count -and
      (Test-IsAcquisitionWrapperToken $commandTokens[$index])) {
    return 'acquisition'
  }
  return Get-AmbiguousOrBenignClassification $commandTokens
}

function Test-IsSafePowerShellCommandElement($Element) {
  if ($Element -is [Management.Automation.Language.StringConstantExpressionAst] -or
      $Element -is [Management.Automation.Language.ConstantExpressionAst]) {
    return $true
  }
  if ($Element -isnot [Management.Automation.Language.CommandParameterAst]) { return $false }
  if ($null -eq $Element.Argument) { return $true }
  return $Element.Argument -is [Management.Automation.Language.StringConstantExpressionAst] -or
    $Element.Argument -is [Management.Automation.Language.ConstantExpressionAst]
}

function Get-SimplePowerShellTokenClassification([string[]]$PayloadTokens) {
  $payload = @($PayloadTokens)
  if ($payload.Count -eq 0) { return '' }
  $targetIndex = if ($payload[0] -ceq '&') { 1 } else { 0 }
  if ($targetIndex -lt $payload.Count -and
      (Test-ExecutableTokenHasEnvironmentSelectedBasename $payload[$targetIndex])) {
    return 'ambiguous'
  }
  if ($targetIndex -ge $payload.Count -or
      -not (Test-IsAcquisitionWrapperToken $payload[$targetIndex])) {
    return ''
  }
  foreach ($token in $payload[$targetIndex..($payload.Count - 1)]) {
    if ($token -match '(?:&&|\|\||[;&|<>`\r\n]|\$\(|[{}])') { return 'ambiguous' }
  }
  return 'acquisition'
}

function Get-StartProcessTargetText($Command) {
  $elements = @($Command.CommandElements)
  for ($index = 1; $index -lt $elements.Count; $index++) {
    $element = $elements[$index]
    if ($element -is [Management.Automation.Language.CommandParameterAst] -and
        $element.ParameterName -ieq 'FilePath') {
      if ($null -ne $element.Argument) { return $element.Argument.Extent.Text }
      if ($index + 1 -lt $elements.Count) { return $elements[$index + 1].Extent.Text }
      return ''
    }
  }
  if ($elements.Count -gt 1 -and
      $elements[1].Extent.Text -match '(?i)^-FilePath(?::|=)(?<value>.*)$') {
    return $Matches.value
  }
  if ($elements.Count -gt 1 -and
      $elements[1] -isnot [Management.Automation.Language.CommandParameterAst]) {
    return $elements[1].Extent.Text
  }
  return ''
}

function Get-StartProcessTargetElement($Command) {
  $elements = @($Command.CommandElements)
  for ($index = 1; $index -lt $elements.Count; $index++) {
    $element = $elements[$index]
    if ($element -is [Management.Automation.Language.CommandParameterAst] -and
        $element.ParameterName -ieq 'FilePath') {
      if ($null -ne $element.Argument) { return $element.Argument }
      if ($index + 1 -lt $elements.Count) { return $elements[$index + 1] }
      return $null
    }
  }
  if ($elements.Count -gt 1 -and
      $elements[1] -isnot [Management.Automation.Language.CommandParameterAst]) {
    return $elements[1]
  }
  return $null
}

function Test-IsStaticallyResolvedPowerShellElement($Element) {
  if ($null -eq $Element) { return $false }
  if ($Element -is [Management.Automation.Language.StringConstantExpressionAst] -or
      $Element -is [Management.Automation.Language.ConstantExpressionAst]) {
    return $true
  }
  if ($Element -is [Management.Automation.Language.ExpandableStringExpressionAst]) {
    return @($Element.NestedExpressions).Count -eq 0
  }
  return $false
}

function Test-HasDynamicPowerShellExecutionTarget($Command) {
  $elements = @($Command.CommandElements)
  if ($Command.InvocationOperator -in @(
      [Management.Automation.Language.TokenKind]::Ampersand,
      [Management.Automation.Language.TokenKind]::Dot
    )) {
    return $elements.Count -eq 0 -or
      -not (Test-IsStaticallyResolvedPowerShellElement $elements[0])
  }

  $commandName = ([string]$Command.GetCommandName()).ToLowerInvariant()
  if ($commandName -in @('invoke-expression', 'iex')) {
    if ($elements.Count -lt 2) { return $false }
    return @($elements | Select-Object -Skip 1 | Where-Object {
      -not (Test-IsStaticallyResolvedPowerShellElement $_)
    }).Count -gt 0
  }
  if ($commandName -in @('start-process', 'saps', 'start')) {
    $target = Get-StartProcessTargetElement $Command
    return $null -ne $target -and
      -not (Test-IsStaticallyResolvedPowerShellElement $target)
  }
  return $false
}

function Test-IsIndirectPowerShellAcquisition($Command) {
  $commandName = ([string]$Command.GetCommandName()).ToLowerInvariant()
  if ($commandName -in @('invoke-expression', 'iex')) {
    $argumentText = @($Command.CommandElements | Select-Object -Skip 1 | ForEach-Object {
      $_.Extent.Text
    }) -join ' '
    return Test-TextContainsBoundedAcquisitionIndicator $argumentText
  }
  if ($commandName -in @('start-process', 'saps', 'start')) {
    $targetText = Get-StartProcessTargetText $Command
    if ($targetText) {
      if ($targetText.Length -ge 2 -and
          (($targetText.StartsWith("'") -and $targetText.EndsWith("'")) -or
           ($targetText.StartsWith('"') -and $targetText.EndsWith('"')))) {
        $targetText = $targetText.Substring(1, $targetText.Length - 2)
      }
      return Test-IsAcquisitionWrapperToken $targetText
    }
    $argumentText = @($Command.CommandElements | Select-Object -Skip 1 | ForEach-Object {
      $_.Extent.Text
    }) -join ' '
    return Test-TextContainsBoundedAcquisitionIndicator $argumentText
  }
  return $false
}

function Get-IndirectPowerShellTokenClassification([string[]]$PayloadTokens) {
  $payload = @($PayloadTokens)
  if ($payload.Count -lt 2) { return '' }
  $commandName = $payload[0].ToLowerInvariant()
  if ($commandName -in @('invoke-expression', 'iex')) {
    if (Test-TokensContainAcquisitionIndicator $payload[1..($payload.Count - 1)]) {
      return 'ambiguous'
    }
    return ''
  }
  if ($commandName -notin @('start-process', 'saps', 'start')) { return '' }

  $targetToken = ''
  $attachedFilePath = $false
  $targetWasSingleQuotedLiteral = $false
  for ($index = 1; $index -lt $payload.Count; $index++) {
    if ($payload[$index] -ieq '-FilePath') {
      if ($index + 1 -lt $payload.Count) { $targetToken = $payload[$index + 1] }
      break
    }
    if ($payload[$index] -match '(?i)^-FilePath(?::|=)(?<value>.*)$') {
      $targetToken = $Matches.value
      $attachedFilePath = $true
      break
    }
  }
  if (-not $targetToken -and -not $payload[1].StartsWith('-')) {
    $targetToken = $payload[1]
  }
  $targetWasSingleQuotedLiteral = $targetToken.Length -ge 2 -and
    $targetToken.StartsWith("'") -and $targetToken.EndsWith("'")
  if ($targetToken.Length -ge 2 -and
      (($targetToken.StartsWith("'") -and $targetToken.EndsWith("'")) -or
       ($targetToken.StartsWith('"') -and $targetToken.EndsWith('"')))) {
    $targetToken = $targetToken.Substring(1, $targetToken.Length - 2)
  }
  if ($targetToken -and -not $targetWasSingleQuotedLiteral -and
      (Test-ExecutableTokenHasEnvironmentSelectedBasename $targetToken)) {
    return 'ambiguous'
  }
  if ($targetToken -and (Test-IsAcquisitionWrapperToken $targetToken)) {
    return 'ambiguous'
  }
  if ($attachedFilePath) {
    if (-not $targetWasSingleQuotedLiteral) {
      if ($targetToken -match '^\s*(?:[({]|@\()') { return 'ambiguous' }
      for ($characterIndex = 0; $characterIndex -lt $targetToken.Length; $characterIndex++) {
        if ($targetToken[$characterIndex] -eq '`') {
          $characterIndex++
          continue
        }
        if ($targetToken[$characterIndex] -eq '$') { return 'ambiguous' }
      }
    }
    return 'benign'
  }
  return ''
}

function Test-PowerShellPayloadContainsComposition([string[]]$PayloadTokens) {
  $payload = @($PayloadTokens)
  $inSingleQuotedString = $false
  for ($index = 0; $index -lt $payload.Count; $index++) {
    $token = $payload[$index]
    $executableText = [Text.StringBuilder]::new()
    for ($characterIndex = 0; $characterIndex -lt $token.Length; $characterIndex++) {
      $character = $token[$characterIndex]
      if ($character -eq "'") {
        if ($inSingleQuotedString -and $characterIndex + 1 -lt $token.Length -and
            $token[$characterIndex + 1] -eq "'") {
          $characterIndex++
          continue
        }
        $inSingleQuotedString = -not $inSingleQuotedString
        continue
      }
      if (-not $inSingleQuotedString) { [void]$executableText.Append($character) }
    }
    $executionToken = $executableText.ToString()
    if ($index -eq 0 -and $executionToken -ceq '.') { return $true }
    if ($executionToken.Contains('$(') -or $executionToken.Contains('@(')) { return $true }
    if ($executionToken -match '(?:&&|\|\||[;|`\r\n])' -or
        $executionToken -match '^(?:\d+|\*)?(?:>>?|<)(?:&\d+)?$') {
      return $true
    }
    if ($executionToken -ceq '&' -and $index -ne 0) { return $true }
  }
  return $false
}

function Get-PowerShellAstCommandTokens($Command) {
  $tokens = [Collections.Generic.List[string]]::new()
  foreach ($element in @($Command.CommandElements)) {
    if ($element -is [Management.Automation.Language.StringConstantExpressionAst]) {
      $tokens.Add([string]$element.Value)
    } else {
      $tokens.Add([string]$element.Extent.Text)
    }
  }
  return @($tokens)
}

function Get-PowerShellCommandTextClassification([string]$CommandText) {
  $parseTokens = $null
  $parseErrors = $null
  $ast = [Management.Automation.Language.Parser]::ParseInput(
    $CommandText,
    [ref]$parseTokens,
    [ref]$parseErrors
  )
  if (@($parseErrors).Count -eq 0) {
    $statements = @($ast.EndBlock.Statements)
    if ($statements.Count -eq 1 -and
        $statements[0] -is [Management.Automation.Language.PipelineAst] -and
        -not $statements[0].Background -and
        $statements[0].PipelineElements.Count -eq 1) {
      $command = $statements[0].PipelineElements[0]
      $safeInvocationOperator = $command.InvocationOperator -in @(
        [Management.Automation.Language.TokenKind]::Unknown,
        [Management.Automation.Language.TokenKind]::Ampersand
      )
      $safeElements = @($command.CommandElements | Where-Object {
        -not (Test-IsSafePowerShellCommandElement $_)
      }).Count -eq 0
      if ($command -is [Management.Automation.Language.CommandAst] -and
          $command.Redirections.Count -eq 0 -and
          $safeInvocationOperator -and
          $safeElements) {
        if (Test-IsAcquisitionWrapperToken $command.GetCommandName()) {
          return 'acquisition'
        }
        $safeCommandTokens = @(Get-PowerShellAstCommandTokens $command)
        if ($safeCommandTokens.Count -gt 0 -and
            (Test-IsPythonLauncherStem (Get-CommandTokenStem $safeCommandTokens[0]))) {
          $safeCommandClassification = Get-AcquisitionTokenClassification $safeCommandTokens
          if ($safeCommandClassification -in @('acquisition', 'ambiguous')) {
            return $safeCommandClassification
          }
        }
      }
    }
    $commands = @($ast.FindAll({
      param($node)
      $node -is [Management.Automation.Language.CommandAst]
    }, $true))
    foreach ($parsedCommand in $commands) {
      if (Test-HasDynamicPowerShellExecutionTarget $parsedCommand) { return 'ambiguous' }
      if (Test-IsIndirectPowerShellAcquisition $parsedCommand) { return 'ambiguous' }
      $parsedTokens = @(Get-PowerShellAstCommandTokens $parsedCommand)
      if ($parsedTokens.Count -gt 0 -and
          (Get-AcquisitionTokenClassification $parsedTokens) -eq 'acquisition') {
        return 'ambiguous'
      }
      $commandName = $parsedCommand.GetCommandName()
      $commandPositionText = if ($commandName) {
        $commandName
      } elseif ($parsedCommand.CommandElements.Count -gt 0) {
        $parsedCommand.CommandElements[0].Extent.Text
      } else {
        ''
      }
      if (Test-TextContainsBoundedAcquisitionIndicator $commandPositionText) {
        return 'ambiguous'
      }
    }
    return 'benign'
  }
  if (Test-TextContainsBoundedAcquisitionIndicator $CommandText) { return 'ambiguous' }
  return 'benign'
}

function Get-PowerShellCommandPayloadClassification([string[]]$PayloadTokens) {
  $payload = @($PayloadTokens)
  if ($payload.Count -eq 0) { return 'benign' }
  if ((Test-PowerShellPayloadContainsComposition $payload) -and
      (Test-TokensContainExactAcquisitionIndicator $payload)) {
    return 'ambiguous'
  }
  $tokenClassification = Get-SimplePowerShellTokenClassification $payload
  if ($tokenClassification) { return $tokenClassification }
  $indirectClassification = Get-IndirectPowerShellTokenClassification $payload
  if ($indirectClassification) { return $indirectClassification }
  $commandText = $payload -join ' '
  return Get-PowerShellCommandTextClassification $commandText
}

function Get-PowerShellEncodedCommandClassification([string]$EncodedCommand) {
  try {
    $bytes = [Convert]::FromBase64String($EncodedCommand)
    if ($bytes.Length % 2 -ne 0) { return 'ambiguous' }
    $strictUtf16Le = [Text.UnicodeEncoding]::new($false, $false, $true)
    $decodedCommand = $strictUtf16Le.GetString($bytes)
  } catch {
    return 'ambiguous'
  }
  return Get-PowerShellCommandTextClassification $decodedCommand
}

function Get-PythonAcquisitionClassification([string[]]$Tokens, [string]$LauncherStem) {
  $continuingOptions = @('-B', '-d', '-E', '-i', '-I', '-P', '-q', '-R', '-s', '-S', '-u', '-x')
  for ($index = 1; $index -lt $Tokens.Count; $index++) {
    $token = $Tokens[$index]
    if ($token -ceq '-m' -or $token -cmatch '^-m(?<module>.+)$') {
      $attachedModule = $token -cne '-m'
      if (-not $attachedModule -and $index + 1 -ge $Tokens.Count) { return 'benign' }
      $module = if ($attachedModule) { $Matches.module } else { $Tokens[$index + 1] }
      $nextIndex = if ($attachedModule) { $index + 1 } else { $index + 2 }
      if ($module -cin @('applypilot', 'applypilot.cli')) {
        if ($nextIndex -lt $Tokens.Count -and $Tokens[$nextIndex] -ceq 'apply') {
          return 'acquisition'
        }
        return 'benign'
      }
      if ($script:AcquisitionPythonModules -ccontains $module) { return 'acquisition' }
      return 'benign'
    }
    if ($token -ceq '--') {
      if ($index + 1 -ge $Tokens.Count) { return 'benign' }
      $scriptToken = $Tokens[$index + 1]
      if (Test-IsAcquisitionPythonScript $scriptToken) { return 'acquisition' }
      if (Test-TextContainsBoundedAcquisitionIndicator $scriptToken) { return 'ambiguous' }
      return 'benign'
    }
    if ($token -ceq '-') { return 'ambiguous' }
    if ($token -ceq '-c' -or $token -cmatch '^-c.+$') { return 'ambiguous' }
    if ($token -cin @('-h', '--help', '-V', '--version')) { return 'benign' }
    if ($token -cmatch '^-b{1,2}$') { continue }
    if ($token -cmatch '^-O{1,2}$') { continue }
    if ($token -cmatch '^-v+$') { continue }
    if ($continuingOptions -ccontains $token) { continue }
    if ($token -cin @('-W', '-X')) {
      if ($index + 1 -ge $Tokens.Count) { return 'benign' }
      $index++
      continue
    }
    if ($token -cmatch '^-(?:W|X).+') { continue }
    if ($token -ceq '--check-hash-based-pycs') {
      if ($index + 1 -ge $Tokens.Count -or
          $Tokens[$index + 1] -cnotin @('always', 'default', 'never')) {
        return Get-AmbiguousOrBenignClassification $Tokens[$index..($Tokens.Count - 1)]
      }
      $index++
      continue
    }
    if ($LauncherStem -in @('py', 'pyw') -and $token -match '^-(?:\d(?:\.\d+)?t?|V:.+)$') { continue }
    if ($token.StartsWith('-')) {
      return Get-AmbiguousOrBenignClassification $Tokens[$index..($Tokens.Count - 1)]
    }
    if (Test-IsAcquisitionPythonScript $token) { return 'acquisition' }
    return 'benign'
  }
  return 'ambiguous'
}

function Get-PowerShellOptionMetadata([string]$Token) {
  if (-not $Token.StartsWith('-') -or $Token.StartsWith('--')) { return $null }
  $name = $Token.Substring(1).ToLowerInvariant()
  $specifications = @(
    [pscustomobject]@{ name = 'commandwithargs'; minimum = 15; kind = 'command'; aliases = @('cwa') },
    [pscustomobject]@{ name = 'encodedcommand'; minimum = 1; kind = 'encoded'; aliases = @('ec') },
    [pscustomobject]@{ name = 'command'; minimum = 1; kind = 'command'; aliases = @() },
    [pscustomobject]@{ name = 'file'; minimum = 1; kind = 'file'; aliases = @() },
    [pscustomobject]@{ name = 'noprofile'; minimum = 3; kind = 'switch'; aliases = @('nop') },
    [pscustomobject]@{ name = 'nologo'; minimum = 3; kind = 'switch'; aliases = @() },
    [pscustomobject]@{ name = 'noexit'; minimum = 3; kind = 'switch'; aliases = @() },
    [pscustomobject]@{ name = 'noninteractive'; minimum = 4; kind = 'switch'; aliases = @() },
    [pscustomobject]@{ name = 'sshservermode'; minimum = 4; kind = 'opaque'; aliases = @() },
    [pscustomobject]@{ name = 'sta'; minimum = 1; kind = 'switch'; aliases = @() },
    [pscustomobject]@{ name = 'mta'; minimum = 1; kind = 'switch'; aliases = @() },
    [pscustomobject]@{ name = 'executionpolicy'; minimum = 2; kind = 'value'; aliases = @('ep', 'ex') },
    [pscustomobject]@{ name = 'inputformat'; minimum = 2; kind = 'value'; aliases = @() },
    [pscustomobject]@{ name = 'outputformat'; minimum = 2; kind = 'value'; aliases = @() },
    [pscustomobject]@{ name = 'windowstyle'; minimum = 2; kind = 'value'; aliases = @() },
    [pscustomobject]@{ name = 'workingdirectory'; minimum = 2; kind = 'value'; aliases = @() },
    [pscustomobject]@{ name = 'configurationname'; minimum = 3; kind = 'value'; aliases = @() }
  )
  foreach ($specification in $specifications) {
    if ($specification.aliases -contains $name) { return $specification }
    if ($name.Length -ge $specification.minimum -and
        $name.Length -le $specification.name.Length -and
        $specification.name.StartsWith($name, [StringComparison]::OrdinalIgnoreCase)) {
      return $specification
    }
  }
  return $null
}

function Get-PowerShellTerminalClassification(
  [string]$Kind,
  [string[]]$Tokens,
  [int]$ValueIndex
) {
  if ($ValueIndex -ge $Tokens.Count) {
    if ($Kind -eq 'encoded') { return 'ambiguous' }
    return 'benign'
  }
  if ($Kind -eq 'encoded') {
    return Get-PowerShellEncodedCommandClassification $Tokens[$ValueIndex]
  }
  if ($Kind -eq 'command') {
    $payload = @($Tokens[$ValueIndex..($Tokens.Count - 1)])
    if ($payload.Count -gt 0 -and $payload[0] -ceq '-') { return 'ambiguous' }
    return Get-PowerShellCommandPayloadClassification $payload
  }
  $target = $Tokens[$ValueIndex]
  if ($target -ceq '-') { return 'ambiguous' }
  if (Test-ExecutableTokenHasEnvironmentSelectedBasename $target) { return 'ambiguous' }
  if (Test-IsAcquisitionWrapperToken $target) { return 'acquisition' }
  return 'benign'
}

function Get-LatePowerShellTerminalClassification([string[]]$Tokens, [int]$StartIndex) {
  for ($scanIndex = $StartIndex; $scanIndex -lt $Tokens.Count; $scanIndex++) {
    $metadata = Get-PowerShellOptionMetadata $Tokens[$scanIndex]
    if ($null -eq $metadata) { continue }
    if ($metadata.kind -in @('command', 'encoded', 'file')) {
      return Get-PowerShellTerminalClassification `
        -Kind $metadata.kind `
        -Tokens $Tokens `
        -ValueIndex ($scanIndex + 1)
    }
    if ($metadata.kind -eq 'opaque') { return 'ambiguous' }
    if ($metadata.kind -eq 'value') { $scanIndex++ }
  }
  return $null
}

function Get-PowerShellAcquisitionClassification([string[]]$Tokens) {
  for ($index = 1; $index -lt $Tokens.Count; $index++) {
    $token = $Tokens[$index]
    if ($token -ceq '-') { return 'ambiguous' }
    if ($token -eq '--') { return 'benign' }
    $metadata = Get-PowerShellOptionMetadata $token
    if ($null -ne $metadata -and $metadata.kind -in @('command', 'encoded', 'file')) {
      return Get-PowerShellTerminalClassification `
        -Kind $metadata.kind `
        -Tokens $Tokens `
        -ValueIndex ($index + 1)
    }
    if ($null -ne $metadata -and $metadata.kind -eq 'opaque') { return 'ambiguous' }
    if ($null -ne $metadata -and $metadata.kind -eq 'switch') { continue }
    if ($null -ne $metadata -and $metadata.kind -eq 'value') {
      if ($index + 1 -ge $Tokens.Count) { return 'benign' }
      $index++
      continue
    }
    if ($token.StartsWith('-')) {
      $lateClassification = Get-LatePowerShellTerminalClassification `
        -Tokens $Tokens `
        -StartIndex ($index + 1)
      if ($lateClassification -in @('acquisition', 'ambiguous')) { return 'ambiguous' }
      return Get-AmbiguousOrBenignClassification $Tokens[$index..($Tokens.Count - 1)]
    }
    if (Test-IsAcquisitionWrapperToken $Tokens[$index]) { return 'acquisition' }
    return 'benign'
  }
  return 'ambiguous'
}

function Test-CmdTokensContainControlSyntax([string[]]$Tokens) {
  foreach ($token in @($Tokens)) {
    if ($token -match '(?:&&|\|\||[&|<>^()\r\n])') { return $true }
  }
  return $false
}

function ConvertFrom-CmdCaretLayer([string]$Token) {
  $result = [Text.StringBuilder]::new()
  for ($index = 0; $index -lt $Token.Length; $index++) {
    if ($Token[$index] -eq '^' -and $index + 1 -lt $Token.Length) {
      $index++
    }
    [void]$result.Append($Token[$index])
  }
  return $result.ToString()
}

function Get-CmdPostModeClassification(
  [string[]]$PayloadTokens,
  [int]$Depth = 0
) {
  $payload = @($PayloadTokens)
  if ($payload.Count -eq 0) { return 'benign' }
  $payloadText = $payload -join ' '
  if ($payloadText -match '(?:%[^%\r\n]+%|![^!\r\n]+!)' -and
      (Test-CmdTokensContainControlSyntax $payload)) {
    return 'ambiguous'
  }
  if (Test-CmdTokensContainControlSyntax $payload) {
    $unescapedPayload = @($payload | ForEach-Object { $_ -replace '\^(.)', '$1' })
    if (Test-TokensContainExactAcquisitionIndicator $unescapedPayload) { return 'ambiguous' }
  }
  $containsExactIndicator = Test-TokensContainExactAcquisitionIndicator $payload

  $index = 0
  $current = $payload[$index]
  if ($current -ceq '@') {
    $index++
    if ($index -ge $payload.Count) { return 'benign' }
    $current = $payload[$index]
  } elseif ($current.StartsWith('@')) {
    $current = $current.Substring(1).TrimStart()
    if (-not $current) {
      $index++
      if ($index -ge $payload.Count) { return 'benign' }
      $current = $payload[$index]
    }
  }

  if ($current -ieq 'call') {
    $index++
    if ($index -ge $payload.Count) {
      if ($containsExactIndicator) { return 'ambiguous' }
      return 'benign'
    }
    $candidate = $payload[$index]
  } elseif ($current -match '(?i)^call\s+(?<target>.+)$') {
    $candidate = $Matches.target.Trim()
  } else {
    $candidate = $current
  }

  if (Test-ExecutableTokenHasEnvironmentSelectedBasename $candidate) {
    return 'ambiguous'
  }

  if ((Get-CommandTokenStem $candidate) -eq 'cmd') {
    if ($Depth -ge 4) { return 'ambiguous' }
    $nestedTokens = @($payload[$index..($payload.Count - 1)] | ForEach-Object {
      ConvertFrom-CmdCaretLayer $_
    })
    return Get-CmdAcquisitionClassification -Tokens $nestedTokens -Depth ($Depth + 1)
  }

  $candidateIsAcquisitionWrapper = Test-IsAcquisitionWrapperToken $candidate
  if ($payloadText -match '(?:%[^%\r\n]+%|![^!\r\n]+!)' -and
      -not $candidateIsAcquisitionWrapper) {
    return 'ambiguous'
  }

  $trailingTokens = if ($index + 1 -lt $payload.Count) {
    @($payload[($index + 1)..($payload.Count - 1)])
  } else {
    @()
  }
  if ($candidateIsAcquisitionWrapper) {
    if (Test-CmdTokensContainControlSyntax $trailingTokens) { return 'ambiguous' }
    return 'acquisition'
  }
  $candidateStem = Get-CommandTokenStem $candidate
  if (Test-IsPythonLauncherStem $candidateStem) {
    return Get-PythonAcquisitionClassification `
      -Tokens (@($candidate) + $trailingTokens) `
      -LauncherStem $candidateStem
  }
  if ($containsExactIndicator) { return 'ambiguous' }
  return 'benign'
}

function Get-CmdSwitchTokenParse([string]$RawToken) {
  $position = 0
  while ($position -lt $RawToken.Length) {
    if ($RawToken[$position] -ne '/') {
      return [ordered]@{ state = 'unfamiliar'; payload = $null }
    }
    $position++
    if ($position -ge $RawToken.Length) {
      return [ordered]@{ state = 'unfamiliar'; payload = $null }
    }

    $option = [char]::ToLowerInvariant($RawToken[$position])
    if ($option -in @('c', 'k', 'r')) {
      return [ordered]@{
        state = 'terminal'
        payload = $RawToken.Substring($position + 1)
      }
    }
    if ($option -in @('d', 's', 'q', 'a', 'u', 'x', 'y')) {
      $position++
    } elseif ($option -eq 't') {
      $remaining = $RawToken.Substring($position)
      if ($remaining -notmatch '^(?i:t:[0-9a-f]{2})') {
        return [ordered]@{ state = 'unfamiliar'; payload = $null }
      }
      $position += $Matches[0].Length
    } elseif ($option -in @('e', 'f', 'v')) {
      $remaining = $RawToken.Substring($position)
      if ($remaining -notmatch '^(?i:(?:e|f|v):(?:on|off))') {
        return [ordered]@{ state = 'unfamiliar'; payload = $null }
      }
      $position += $Matches[0].Length
    } else {
      return [ordered]@{ state = 'unfamiliar'; payload = $null }
    }

    if ($position -lt $RawToken.Length -and $RawToken[$position] -ne '/') {
      return [ordered]@{ state = 'unfamiliar'; payload = $null }
    }
  }
  return [ordered]@{ state = 'continuing'; payload = $null }
}

function Get-CmdUnfamiliarClassification([string[]]$Tokens) {
  $tokenText = @($Tokens) -join ' '
  if ($tokenText -match '(?:%[^%\r\n]+%|![^!\r\n]+!)') { return 'ambiguous' }
  if (Test-CmdTokensContainControlSyntax $Tokens) {
    $unescapedTokens = @($Tokens | ForEach-Object { $_ -replace '\^(.)', '$1' })
    if (Test-TokensContainExactAcquisitionIndicator $unescapedTokens) { return 'ambiguous' }
  }
  if ((Get-AmbiguousOrBenignClassification $Tokens) -eq 'ambiguous') { return 'ambiguous' }
  foreach ($token in @($Tokens)) {
    foreach ($match in [regex]::Matches($token, '(?i)/(?:c|k|r)(?<payload>[^/]*)')) {
      if (Test-TextContainsBoundedAcquisitionIndicator $match.Groups['payload'].Value) {
        return 'ambiguous'
      }
    }
  }
  return 'benign'
}

function Get-CmdAcquisitionClassification([string[]]$Tokens, [int]$Depth = 0) {
  for ($index = 1; $index -lt $Tokens.Count; $index++) {
    $rawToken = $Tokens[$index]
    if (-not $rawToken.StartsWith('/')) {
      return Get-CmdUnfamiliarClassification $Tokens[$index..($Tokens.Count - 1)]
    }
    $parsed = Get-CmdSwitchTokenParse $rawToken
    if ($parsed.state -eq 'continuing') { continue }
    if ($parsed.state -eq 'terminal') {
      $payload = @()
      if ($parsed.payload) { $payload += [string]$parsed.payload }
      if ($index + 1 -lt $Tokens.Count) {
        $payload += $Tokens[($index + 1)..($Tokens.Count - 1)]
      }
      return Get-CmdPostModeClassification -PayloadTokens $payload -Depth $Depth
    }
    return Get-CmdUnfamiliarClassification $Tokens[$index..($Tokens.Count - 1)]
  }
  return 'ambiguous'
}

function Get-AcquisitionTokenClassification([string[]]$Tokens) {
  $tokens = @($Tokens)
  if ($tokens.Count -eq 0) { return 'benign' }
  if (Test-ExecutableTokenHasEnvironmentSelectedBasename $tokens[0]) { return 'ambiguous' }
  $launcher = Get-CommandTokenStem $tokens[0]
  if ($launcher -eq 'applypilot') {
    if ($tokens.Count -gt 1 -and $tokens[1] -ceq 'apply') { return 'acquisition' }
    return 'benign'
  }
  if ($script:AcquisitionConsoleBasenames -contains $launcher) { return 'acquisition' }
  if (Test-IsPythonLauncherStem $launcher) {
    return Get-PythonAcquisitionClassification -Tokens $tokens -LauncherStem $launcher
  }
  if ($launcher -in @('pwsh', 'powershell')) { return Get-PowerShellAcquisitionClassification $tokens }
  if ($launcher -eq 'cmd') { return Get-CmdAcquisitionClassification $tokens }
  if (Test-IsAcquisitionWrapperToken $tokens[0]) { return 'acquisition' }
  return 'benign'
}

function Get-CommandLineClassification([string]$CommandLine) {
  if ([string]::IsNullOrWhiteSpace($CommandLine)) { return 'benign' }
  try {
    $tokens = @(ConvertFrom-DeterministicCommandLine $CommandLine)
  } catch {
    if (Test-CommandTextContainsAcquisitionIndicator $CommandLine) { return 'ambiguous' }
    return 'benign'
  }
  return Get-AcquisitionTokenClassification $tokens
}

function Test-IsAcquisitionCommandLine([string]$CommandLine) {
  return (Get-CommandLineClassification $CommandLine) -eq 'acquisition'
}

function Get-TaskActionClassification([string]$Executable, [string]$Arguments) {
  if ([string]::IsNullOrWhiteSpace($Executable)) { return 'benign' }
  try {
    $argumentTokens = if ([string]::IsNullOrWhiteSpace($Arguments)) {
      @()
    } else {
      @(ConvertFrom-DeterministicCommandLine $Arguments)
    }
  } catch {
    if (Test-CommandTextContainsAcquisitionIndicator $Arguments) { return 'ambiguous' }
    return 'benign'
  }
  return Get-AcquisitionTokenClassification (@($Executable) + $argumentTokens)
}

if ($PSCmdlet.ParameterSetName -eq 'Probe') {
  $classification = Get-CommandLineClassification $CommandLineProbe
  [ordered]@{
    mode = 'test'
    operational = $false
    classification = $classification
    matched = $classification -eq 'acquisition'
  } | ConvertTo-Json -Compress
  exit 0
}

function Get-LegacyTaskClassification($Task) {
  if ([string]$Task.TaskName -match $script:KnownAcquisitionTaskNamePattern) { return 'acquisition' }
  $classification = 'benign'
  foreach ($action in @($Task.Actions)) {
    $actionClassification = Get-TaskActionClassification -Executable ([string]$action.Execute) -Arguments ([string]$action.Arguments)
    if ($actionClassification -eq 'acquisition') { return 'acquisition' }
    if ($actionClassification -eq 'ambiguous') { $classification = 'ambiguous' }
  }
  return $classification
}

function Get-LegacyServiceClassification($Service) {
  if ([string]$Service.Name -match $script:KnownAcquisitionServiceNamePattern) { return 'acquisition' }
  return Get-CommandLineClassification ([string]$Service.PathName)
}

# Operational adapters are deliberately narrow and cannot be dynamically replaced.
function Get-LegacyTasks {
  return @(Get-ScheduledTask -ErrorAction Stop | ForEach-Object {
    $classification = Get-LegacyTaskClassification $_
    if ($classification -ne 'benign') {
      $_ | Add-Member -NotePropertyName AuthorityClassification -NotePropertyValue $classification -Force
      $_
    }
  })
}
function Stop-LegacyTask($Task) {
  Stop-ScheduledTask -TaskName $Task.TaskName -TaskPath $Task.TaskPath -ErrorAction Stop
}
function Disable-LegacyTask($Task) {
  Disable-ScheduledTask -TaskName $Task.TaskName -TaskPath $Task.TaskPath -ErrorAction Stop | Out-Null
}
function Get-LegacyServices {
  return @(Get-CimInstance Win32_Service -ErrorAction Stop | ForEach-Object {
      $classification = Get-LegacyServiceClassification $_
      if ($classification -ne 'benign') {
      [pscustomobject]@{
        Name = [string]$_.Name
        Status = [string]$_.State
        StartType = switch ([string]$_.StartMode) {
          'Auto' { 'Automatic' }
          'Manual' { 'Manual' }
          'Disabled' { 'Disabled' }
          default { [string]$_.StartMode }
        }
        PathName = [string]$_.PathName
        AuthorityClassification = $classification
      }
      }
    })
}
function Stop-LegacyService($Service) {
  Stop-Service -Name $Service.Name -Force -ErrorAction Stop
}
function Disable-LegacyService($Service) {
  Set-Service -Name $Service.Name -StartupType Disabled -ErrorAction Stop
}
function Get-LegacyProcesses {
  $currentPid = $PID
  return @(Get-CimInstance Win32_Process -ErrorAction Stop | ForEach-Object {
    if ($_.ProcessId -ne $currentPid) {
      $classification = Get-CommandLineClassification ([string]$_.CommandLine)
      if ($classification -ne 'benign') {
        $_ | Add-Member -NotePropertyName AuthorityClassification -NotePropertyValue $classification -Force
        $_
      }
    }
  })
}

function Initialize-LegacyProcessNativeApi {
  if ('ApplyPilot.LegacyProcessNativeApi' -as [type]) { return }
  Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace ApplyPilot
{
    public static class LegacyProcessNativeApi
    {
        private const uint ProcessTerminate = 0x0001;
        private const uint ProcessQueryLimitedInformation = 0x1000;

        [StructLayout(LayoutKind.Sequential)]
        private struct FileTime
        {
            public uint Low;
            public uint High;

            public long ToInt64()
            {
                return ((long)High << 32) | Low;
            }
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern SafeProcessHandle OpenProcess(
            uint desiredAccess,
            bool inheritHandle,
            int processId);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetProcessTimes(
            SafeProcessHandle process,
            out FileTime creationTime,
            out FileTime exitTime,
            out FileTime kernelTime,
            out FileTime userTime);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool QueryFullProcessImageName(
            SafeProcessHandle process,
            uint flags,
            StringBuilder imagePath,
            ref uint size);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateProcess(SafeProcessHandle process, uint exitCode);

        public static SafeProcessHandle Open(int processId)
        {
            SafeProcessHandle handle = OpenProcess(
                ProcessTerminate | ProcessQueryLimitedInformation,
                false,
                processId);
            if (handle == null || handle.IsInvalid)
            {
                if (handle != null) handle.Dispose();
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            return handle;
        }

        public static long GetCreationFileTimeUtc(SafeProcessHandle process)
        {
            FileTime creation;
            FileTime exit;
            FileTime kernel;
            FileTime user;
            if (!GetProcessTimes(process, out creation, out exit, out kernel, out user))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return creation.ToInt64();
        }

        public static string GetImagePath(SafeProcessHandle process)
        {
            uint size = 32768;
            StringBuilder imagePath = new StringBuilder((int)size);
            if (!QueryFullProcessImageName(process, 0, imagePath, ref size))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return imagePath.ToString();
        }

        public static void Terminate(SafeProcessHandle process)
        {
            if (!TerminateProcess(process, 1))
                throw new Win32Exception(Marshal.GetLastWin32Error());
        }
    }
}
'@
}

function Open-LegacyProcessHandle([int]$ProcessId) {
  Initialize-LegacyProcessNativeApi
  return [ApplyPilot.LegacyProcessNativeApi]::Open($ProcessId)
}

function Get-LegacyProcessHandleIdentity($Handle) {
  return [ordered]@{
    creation_file_time_utc = [ApplyPilot.LegacyProcessNativeApi]::GetCreationFileTimeUtc($Handle)
    executable_path = [ApplyPilot.LegacyProcessNativeApi]::GetImagePath($Handle)
  }
}

function Stop-LegacyProcessHandle($Handle) {
  [ApplyPilot.LegacyProcessNativeApi]::Terminate($Handle)
}

function ConvertTo-ProcessCreationFileTimeUtc($CreationDate) {
  if ($CreationDate -is [datetimeoffset]) {
    return $CreationDate.UtcDateTime.ToFileTimeUtc()
  }
  if ($CreationDate -is [datetime]) {
    return $CreationDate.ToUniversalTime().ToFileTimeUtc()
  }
  $text = [string]$CreationDate
  if ([string]::IsNullOrWhiteSpace($text)) { throw 'missing process creation time' }
  try {
    return [Management.ManagementDateTimeConverter]::ToDateTime($text).ToUniversalTime().ToFileTimeUtc()
  } catch {
    return [datetime]::Parse(
      $text,
      [Globalization.CultureInfo]::InvariantCulture,
      [Globalization.DateTimeStyles]::AssumeUniversal
    ).ToUniversalTime().ToFileTimeUtc()
  }
}

function Get-NormalizedExecutablePath([string]$Path) {
  if ([string]::IsNullOrWhiteSpace($Path)) { throw 'missing executable path' }
  return [IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Add-SkippedProcessAction([string]$TargetDigest, [string]$Result) {
  $script:SkippedActions.Add([ordered]@{
    action = 'stop_process'
    target_digest = $TargetDigest
    result = $Result
  })
}

function Stop-LegacyProcess($Process) {
  $targetDigest = Get-ProcessStableIdentityDigest $Process
  $handle = $null
  try {
    $handle = Open-LegacyProcessHandle ([int]$Process.ProcessId)
  } catch {
    Add-SkippedProcessAction -TargetDigest $targetDigest -Result 'handle_open_failed'
    return
  }
  try {
    try {
      $currentIdentity = Get-LegacyProcessHandleIdentity $handle
      $expectedCreation = ConvertTo-ProcessCreationFileTimeUtc $Process.CreationDate
      $expectedExecutable = Get-NormalizedExecutablePath ([string]$Process.ExecutablePath)
      $currentExecutable = Get-NormalizedExecutablePath ([string]$currentIdentity.executable_path)
    } catch {
      Add-SkippedProcessAction -TargetDigest $targetDigest -Result 'identity_unavailable'
      return
    }
    $creationDelta = [Math]::Abs(
      [long]$currentIdentity.creation_file_time_utc - [long]$expectedCreation
    )
    if ($creationDelta -gt 9 -or
        $currentExecutable -ine $expectedExecutable) {
      Add-SkippedProcessAction -TargetDigest $targetDigest -Result 'identity_changed'
      return
    }
    try {
      Stop-LegacyProcessHandle $handle
    } catch {
      Add-SkippedProcessAction -TargetDigest $targetDigest -Result 'termination_failed'
    }
  } finally {
    $handle.Dispose()
  }
}
function Initialize-KnownWrapperNativeApi {
  if ('ApplyPilot.KnownWrapperNativeApi' -as [type]) { return }
  Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace ApplyPilot
{
    public static class KnownWrapperNativeApi
    {
        private const uint GenericRead = 0x80000000;
        private const uint GenericWrite = 0x40000000;
        private const uint ShareRead = 0x00000001;
        private const uint ShareWrite = 0x00000002;
        private const uint ShareDelete = 0x00000004;
        private const uint OpenExisting = 3;
        private const uint FileFlagBackupSemantics = 0x02000000;
        private const uint FileFlagOpenReparsePoint = 0x00200000;
        private const uint FileAttributeDirectory = 0x00000010;
        private const uint FileAttributeReparsePoint = 0x00000400;
        private const uint DriveFixed = 3;
        private const int FileStandardInfoClass = 1;
        private const int FileAttributeTagInfoClass = 9;
        private const int FileIdInfoClass = 18;
        private static readonly Guid LocalAppDataFolderId =
            new Guid("F1B32785-6FBA-4FCF-9D55-7B8E7F157091");

        [StructLayout(LayoutKind.Sequential)]
        private struct FileAttributeTagInfo
        {
            public uint FileAttributes;
            public uint ReparseTag;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct FileStandardInfo
        {
            public long AllocationSize;
            public long EndOfFile;
            public uint NumberOfLinks;
            public byte DeletePending;
            public byte Directory;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct FileId128
        {
            public ulong Low;
            public ulong High;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct FileIdInfo
        {
            public ulong VolumeSerialNumber;
            public FileId128 FileId;
        }

        internal sealed class HandleIdentity
        {
            public ulong VolumeSerialNumber;
            public ulong FileIdLow;
            public ulong FileIdHigh;
            public string FinalPath;
            public string Token;
        }

        internal sealed class HeldComponent
        {
            public SafeFileHandle Handle;
            public HandleIdentity Identity;
        }

        public sealed class WrapperOpenLease : IDisposable
        {
            private readonly List<HeldComponent> components;
            private readonly HandleIdentity leafIdentity;
            private bool disposed;

            internal WrapperOpenLease(
                List<HeldComponent> heldComponents,
                SafeFileHandle leafHandle,
                HandleIdentity identity,
                bool writable)
            {
                components = heldComponents;
                leafIdentity = identity;
                IdentityToken = identity.Token;
                try
                {
                    Stream = new FileStream(
                        leafHandle,
                        writable ? FileAccess.ReadWrite : FileAccess.Read);
                }
                catch
                {
                    leafHandle.Dispose();
                    DisposeComponents();
                    throw;
                }
            }

            public FileStream Stream { get; private set; }
            public string IdentityToken { get; private set; }

            public void Validate()
            {
                if (disposed)
                    throw new ObjectDisposedException(nameof(WrapperOpenLease));
                foreach (HeldComponent component in components)
                {
                    HandleIdentity current = ReadIdentity(component.Handle, true);
                    RequireSameIdentity(component.Identity, current);
                }
                HandleIdentity currentLeaf = ReadIdentity(Stream.SafeFileHandle, false);
                RequireSameIdentity(leafIdentity, currentLeaf);
                string parent = Path.GetDirectoryName(currentLeaf.FinalPath).TrimEnd('\\');
                string heldParent = components[components.Count - 1].Identity.FinalPath.TrimEnd('\\');
                if (!String.Equals(parent, heldParent, StringComparison.OrdinalIgnoreCase))
                    throw new IOException("wrapper parent identity changed");
            }

            private void DisposeComponents()
            {
                for (int index = components.Count - 1; index >= 0; index--)
                    components[index].Handle.Dispose();
                components.Clear();
            }

            public void Dispose()
            {
                if (disposed) return;
                disposed = true;
                try
                {
                    if (Stream != null) Stream.Dispose();
                }
                finally
                {
                    DisposeComponents();
                }
            }
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateFileW(
            string fileName,
            uint desiredAccess,
            uint shareMode,
            IntPtr securityAttributes,
            uint creationDisposition,
            uint flagsAndAttributes,
            IntPtr templateFile);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetFileInformationByHandleEx(
            SafeFileHandle file,
            int informationClass,
            IntPtr information,
            uint bufferSize);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern uint GetFinalPathNameByHandleW(
            SafeFileHandle file,
            StringBuilder filePath,
            uint filePathSize,
            uint flags);

        [DllImport("shell32.dll", SetLastError = false)]
        private static extern int SHGetKnownFolderPath(
            ref Guid folderId,
            uint flags,
            IntPtr token,
            out IntPtr path);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        private static extern uint GetDriveTypeW(string rootPathName);

        private static T GetInformation<T>(SafeFileHandle handle, int informationClass)
            where T : struct
        {
            int size = Marshal.SizeOf<T>();
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try
            {
                if (!GetFileInformationByHandleEx(handle, informationClass, buffer, (uint)size))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                return Marshal.PtrToStructure<T>(buffer);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        private static string GetFinalVolumePath(SafeFileHandle handle)
        {
            var buffer = new StringBuilder(512);
            uint length = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, 0);
            if (length == 0)
                throw new Win32Exception(Marshal.GetLastWin32Error());
            if (length >= buffer.Capacity)
            {
                buffer.Capacity = checked((int)length + 1);
                length = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, 0);
                if (length == 0 || length >= buffer.Capacity)
                    throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            return buffer.ToString().TrimEnd('\\');
        }

        private static string ComputeIdentityToken(FileIdInfo fileId, string finalPath)
        {
            string material = String.Format(
                "{0:X16}|{1:X16}{2:X16}|{3}",
                fileId.VolumeSerialNumber,
                fileId.FileId.Low,
                fileId.FileId.High,
                finalPath.ToUpperInvariant());
            byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(material));
            return Convert.ToHexString(digest).ToLowerInvariant();
        }

        private static HandleIdentity ReadIdentity(SafeFileHandle handle, bool expectDirectory)
        {
            FileAttributeTagInfo tag = GetInformation<FileAttributeTagInfo>(
                handle, FileAttributeTagInfoClass);
            FileStandardInfo standard = GetInformation<FileStandardInfo>(
                handle, FileStandardInfoClass);
            FileIdInfo fileId = GetInformation<FileIdInfo>(handle, FileIdInfoClass);
            bool isDirectory = standard.Directory != 0 ||
                (tag.FileAttributes & FileAttributeDirectory) != 0;
            if (isDirectory != expectDirectory ||
                (tag.FileAttributes & FileAttributeReparsePoint) != 0 ||
                tag.ReparseTag != 0 ||
                (!expectDirectory && standard.NumberOfLinks != 1))
                throw new IOException("wrapper handle identity is invalid");
            string finalPath = GetFinalVolumePath(handle);
            return new HandleIdentity
            {
                VolumeSerialNumber = fileId.VolumeSerialNumber,
                FileIdLow = fileId.FileId.Low,
                FileIdHigh = fileId.FileId.High,
                FinalPath = finalPath,
                Token = ComputeIdentityToken(fileId, finalPath)
            };
        }

        private static void RequireSameIdentity(HandleIdentity expected, HandleIdentity actual)
        {
            if (expected.VolumeSerialNumber != actual.VolumeSerialNumber ||
                expected.FileIdLow != actual.FileIdLow ||
                expected.FileIdHigh != actual.FileIdHigh ||
                !String.Equals(expected.FinalPath, actual.FinalPath, StringComparison.OrdinalIgnoreCase) ||
                !String.Equals(expected.Token, actual.Token, StringComparison.Ordinal))
                throw new IOException("wrapper handle identity changed");
        }

        private static SafeFileHandle OpenDirectoryComponent(string path, bool denyDeleteSharing)
        {
            SafeFileHandle handle = CreateFileW(
                ToExtendedLocalPath(path),
                0,
                ShareRead | ShareWrite | (denyDeleteSharing ? 0 : ShareDelete),
                IntPtr.Zero,
                OpenExisting,
                FileFlagBackupSemantics | FileFlagOpenReparsePoint,
                IntPtr.Zero);
            if (handle.IsInvalid)
            {
                int error = Marshal.GetLastWin32Error();
                handle.Dispose();
                throw new Win32Exception(error);
            }
            return handle;
        }

        private static string ToExtendedLocalPath(string path)
        {
            if (String.IsNullOrWhiteSpace(path) ||
                path.StartsWith(@"\\", StringComparison.Ordinal) ||
                path.StartsWith(@"\\?\", StringComparison.Ordinal) ||
                path.StartsWith(@"\\.\", StringComparison.Ordinal))
                throw new IOException("wrapper path is not a canonical local path");
            string fullPath = Path.GetFullPath(path);
            string root = Path.GetPathRoot(fullPath);
            if (String.IsNullOrEmpty(root) || root.Length != 3 ||
                root[1] != ':' || root[2] != '\\' || !Char.IsLetter(root[0]) ||
                fullPath.IndexOf(':', 2) >= 0)
                throw new IOException("wrapper path is not a canonical local path");
            return @"\\?\" + fullPath;
        }

        private static string GetTrustedLocalVolumeRoot(string path)
        {
            if (path.StartsWith(@"\\", StringComparison.Ordinal) ||
                path.StartsWith(@"\\?\", StringComparison.Ordinal) ||
                path.StartsWith(@"\\.\", StringComparison.Ordinal))
                throw new IOException("wrapper volume path is not local");
            string root = Path.GetPathRoot(path);
            if (String.IsNullOrEmpty(root) || root.Length != 3 ||
                root[1] != ':' || root[2] != '\\' ||
                !Char.IsLetter(root[0]))
                throw new IOException("wrapper volume root is ambiguous");
            if (GetDriveTypeW(root) != DriveFixed)
                throw new IOException("wrapper volume is not a fixed local drive");
            return Char.ToUpperInvariant(root[0]) + @":\";
        }

        private static void ValidateHeldChain(List<HeldComponent> held)
        {
            for (int index = 0; index < held.Count; index++)
            {
                HandleIdentity current = ReadIdentity(held[index].Handle, true);
                RequireSameIdentity(held[index].Identity, current);
                if (index > 0)
                {
                    string actualParent = Path.GetDirectoryName(current.FinalPath).TrimEnd('\\');
                    string heldParent = held[index - 1].Identity.FinalPath.TrimEnd('\\');
                    if (!String.Equals(actualParent, heldParent, StringComparison.OrdinalIgnoreCase))
                        throw new IOException("wrapper component parent identity changed");
                }
            }
        }

        private static List<HeldComponent> OpenComponentChain(
            string authorizedRoot,
            string parent,
            bool denyAuthorizedRootDeleteSharing,
            Action<string> afterComponentOpen)
        {
            string volumeRoot = GetTrustedLocalVolumeRoot(authorizedRoot);
            if (!String.Equals(
                    volumeRoot,
                    GetTrustedLocalVolumeRoot(parent),
                    StringComparison.OrdinalIgnoreCase))
                throw new IOException("wrapper root crosses a local volume boundary");
            var paths = new List<string> { volumeRoot };
            string relative = Path.GetRelativePath(volumeRoot, parent);
            string current = volumeRoot;
            foreach (string segment in relative.Split(Path.DirectorySeparatorChar))
            {
                if (String.IsNullOrEmpty(segment) || segment == "." || segment == "..")
                    throw new IOException("wrapper component path is not canonical");
                current = Path.Combine(current, segment);
                paths.Add(current);
            }
            if (!String.Equals(current.TrimEnd('\\'), authorizedRoot.TrimEnd('\\'),
                    StringComparison.OrdinalIgnoreCase))
                throw new IOException("wrapper authorized root is ambiguous");

            var held = new List<HeldComponent>();
            try
            {
                for (int pathIndex = 0; pathIndex < paths.Count; pathIndex++)
                {
                    string componentPath = paths[pathIndex];
                    SafeFileHandle handle = OpenDirectoryComponent(
                        componentPath,
                        denyAuthorizedRootDeleteSharing && pathIndex == paths.Count - 1);
                    try
                    {
                        HandleIdentity identity = ReadIdentity(handle, true);
                        if (held.Count == 0)
                        {
                            string expectedVolumePath = @"\\?\" + volumeRoot.TrimEnd('\\');
                            if (!String.Equals(identity.FinalPath, expectedVolumePath,
                                    StringComparison.OrdinalIgnoreCase))
                                throw new IOException("wrapper volume root is redirected");
                        }
                        held.Add(new HeldComponent
                        {
                            Handle = handle,
                            Identity = identity
                        });
                        if (afterComponentOpen != null) afterComponentOpen(componentPath);
                        ValidateHeldChain(held);
                    }
                    catch
                    {
                        handle.Dispose();
                        throw;
                    }
                }
                return held;
            }
            catch
            {
                for (int index = held.Count - 1; index >= 0; index--)
                    held[index].Handle.Dispose();
                throw;
            }
        }

        public static string GetLocalAppDataPath()
        {
            IntPtr value = IntPtr.Zero;
            Guid folderId = LocalAppDataFolderId;
            int result = SHGetKnownFolderPath(ref folderId, 0, IntPtr.Zero, out value);
            if (result != 0)
                Marshal.ThrowExceptionForHR(result);
            try
            {
                string path = Marshal.PtrToStringUni(value);
                if (String.IsNullOrWhiteSpace(path))
                    throw new IOException("LocalAppData known folder is unavailable");
                return Path.GetFullPath(path).TrimEnd('\\');
            }
            finally
            {
                if (value != IntPtr.Zero) Marshal.FreeCoTaskMem(value);
            }
        }

        public static WrapperOpenLease OpenValidated(
            string path,
            string[] authorizedRoots,
            string[] authorizedBasenames,
            bool writable,
            string expectedIdentityToken,
            Action<string> afterComponentOpen,
            Action beforeLeafOpen)
        {
            string fullPath = Path.GetFullPath(path);
            if (!String.Equals(path, fullPath, StringComparison.OrdinalIgnoreCase) ||
                path.StartsWith(@"\\?\", StringComparison.Ordinal) ||
                path.StartsWith(@"\\.\", StringComparison.Ordinal))
                throw new IOException("wrapper path is not canonical");
            string basename = Path.GetFileName(path);
            if (basename.EndsWith(".", StringComparison.Ordinal) ||
                basename.EndsWith(" ", StringComparison.Ordinal) ||
                basename.IndexOf(':') >= 0)
                throw new IOException("wrapper basename alias is not authorized");
            var basenames = new HashSet<string>(authorizedBasenames, StringComparer.OrdinalIgnoreCase);
            if (!basenames.Contains(basename))
                throw new IOException("wrapper basename is not authorized");

            string parent = Path.GetDirectoryName(fullPath).TrimEnd('\\');
            string authorizedRoot = null;
            foreach (string candidate in authorizedRoots)
            {
                string normalized = Path.GetFullPath(candidate).TrimEnd('\\');
                if (String.Equals(parent, normalized, StringComparison.OrdinalIgnoreCase))
                {
                    authorizedRoot = normalized;
                    break;
                }
            }
            if (authorizedRoot == null)
                throw new IOException("wrapper root is not authorized");
            if (writable && String.IsNullOrWhiteSpace(expectedIdentityToken))
                throw new IOException("wrapper mutation identity is required");

            List<HeldComponent> components = OpenComponentChain(
                authorizedRoot, parent, writable, afterComponentOpen);
            SafeFileHandle leafHandle = null;
            try
            {
                if (beforeLeafOpen != null) beforeLeafOpen();
                leafHandle = CreateFileW(
                    ToExtendedLocalPath(fullPath),
                    writable ? GenericRead | GenericWrite : GenericRead,
                    0,
                    IntPtr.Zero,
                    OpenExisting,
                    FileFlagOpenReparsePoint,
                    IntPtr.Zero);
                if (leafHandle.IsInvalid)
                {
                    int error = Marshal.GetLastWin32Error();
                    leafHandle.Dispose();
                    leafHandle = null;
                    throw new Win32Exception(error);
                }
                HandleIdentity leafIdentity = ReadIdentity(leafHandle, false);
                string heldParent = components[components.Count - 1].Identity.FinalPath.TrimEnd('\\');
                string leafParent = Path.GetDirectoryName(leafIdentity.FinalPath).TrimEnd('\\');
                if (!String.Equals(leafParent, heldParent, StringComparison.OrdinalIgnoreCase) ||
                    !String.Equals(Path.GetFileName(leafIdentity.FinalPath), basename,
                        StringComparison.OrdinalIgnoreCase))
                    throw new IOException("wrapper final path is not authorized");
                foreach (HeldComponent component in components)
                    RequireSameIdentity(component.Identity, ReadIdentity(component.Handle, true));
                if (writable && !String.Equals(
                        expectedIdentityToken, leafIdentity.Token, StringComparison.Ordinal))
                    throw new IOException("wrapper snapshot identity changed");
                var lease = new WrapperOpenLease(components, leafHandle, leafIdentity, writable);
                leafHandle = null;
                components = null;
                try
                {
                    lease.Validate();
                    return lease;
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }
            finally
            {
                if (leafHandle != null) leafHandle.Dispose();
                if (components != null)
                {
                    for (int index = components.Count - 1; index >= 0; index--)
                        components[index].Handle.Dispose();
                }
            }
        }
    }
}
'@
}

function Get-KnownAuthorizedWrapperRoots {
  if ($script:DefinitionImportMode -and $WrapperRoot) { return @($WrapperRoot) }
  Initialize-KnownWrapperNativeApi
  return @(
    (Join-Path $PSScriptRoot '..\.fleet-logs\_task-wrappers'),
    (Join-Path ([ApplyPilot.KnownWrapperNativeApi]::GetLocalAppDataPath()) 'ApplyPilot\_task-wrappers')
  )
}

function Open-KnownWrapperNative(
  [string]$Path,
  [bool]$Writable,
  [string]$ExpectedIdentityToken = $null
) {
  Initialize-KnownWrapperNativeApi
  $afterComponentOpen = $null
  $beforeLeafOpen = $null
  if ($script:DefinitionImportMode -and $null -ne $script:WrapperAfterComponentOpenHook) {
    $afterComponentOpen = [Action[string]]{
      param([string]$ComponentPath)
      if ($null -ne $script:WrapperAfterComponentOpenHook) {
        & $script:WrapperAfterComponentOpenHook $ComponentPath
      }
    }
  }
  if ($script:DefinitionImportMode -and $null -ne $script:WrapperBeforeLeafOpenHook) {
    $beforeLeafOpen = [Action]{ & $script:WrapperBeforeLeafOpenHook }
  }
  return [ApplyPilot.KnownWrapperNativeApi]::OpenValidated(
    $Path,
    [string[]]@(Get-KnownAuthorizedWrapperRoots),
    [string[]]$script:GeneratedAcquisitionWrapperBasenames,
    $Writable,
    $ExpectedIdentityToken,
    $afterComponentOpen,
    $beforeLeafOpen
  )
}

function Open-KnownWrapperSnapshotExclusive([string]$Path) {
  return Open-KnownWrapperNative -Path $Path -Writable $false
}

function Open-KnownWrapperMutationExclusive(
  [string]$Path,
  [string]$ExpectedIdentityToken
) {
  return Open-KnownWrapperNative `
    -Path $Path `
    -Writable $true `
    -ExpectedIdentityToken $ExpectedIdentityToken
}

function Assert-KnownWrapperMutationHandle($Stream) {
  if ($null -eq $script:ActiveWrapperMutationLease) {
    throw 'wrapper mutation lease is unavailable'
  }
  $script:ActiveWrapperMutationLease.Validate()
}

function Read-KnownWrapperStreamBytes($Stream) {
  $Stream.Position = 0
  $buffer = [IO.MemoryStream]::new()
  try {
    $Stream.CopyTo($buffer)
    return ,$buffer.ToArray()
  } finally {
    $buffer.Dispose()
  }
}

function ConvertFrom-KnownWrapperBytes([byte[]]$Content) {
  if ($Content.Length -eq 0) { return '' }
  $buffer = [IO.MemoryStream]::new($Content, $false)
  $reader = [IO.StreamReader]::new(
    $buffer,
    [Text.UTF8Encoding]::new($false, $true),
    $true
  )
  try {
    return $reader.ReadToEnd()
  } catch [Text.DecoderFallbackException] {
    return [Text.Encoding]::Latin1.GetString($Content)
  } finally {
    $reader.Dispose()
    $buffer.Dispose()
  }
}

function Get-KnownWrapperByteDigest([byte[]]$Content) {
  return [Convert]::ToHexString(
    [Security.Cryptography.SHA256]::HashData($Content)
  ).ToLowerInvariant()
}

function Write-KnownWrapperDenyStub($Stream, [byte[]]$Content) {
  Assert-KnownWrapperMutationHandle $Stream
  $Stream.Position = 0
  $Stream.SetLength(0)
  $Stream.Write($Content, 0, $Content.Length)
  $Stream.Flush($true)
  Assert-KnownWrapperMutationHandle $Stream
}

function Restore-KnownWrapperBytes($Stream, [byte[]]$Content) {
  $Stream.Position = 0
  $Stream.SetLength(0)
  if ($Content.Length -gt 0) { $Stream.Write($Content, 0, $Content.Length) }
  $Stream.Flush($true)
  $restored = Read-KnownWrapperStreamBytes $Stream
  if ($restored.Length -ne $Content.Length -or
      (Get-KnownWrapperByteDigest $restored) -cne (Get-KnownWrapperByteDigest $Content)) {
    throw 'wrapper restoration verification failed'
  }
}

function Get-KnownWrapperDenyStubBytes([string]$Extension) {
  $content = if ($Extension -match '(?i)^\.(?:cmd|bat)$') {
    "@echo off`r`necho ApplyPilot acquisition denied: emergency containment 1>&2`r`nexit /b 78`r`n"
  } else {
    "Write-Error 'ApplyPilot acquisition denied: emergency containment'`nexit 78`n"
  }
  return [Text.UTF8Encoding]::new($false).GetBytes($content)
}

function Test-KnownWrapperDenyStub([byte[]]$Content, [string]$Extension) {
  $expected = Get-KnownWrapperDenyStubBytes $Extension
  return $Content.Length -eq $expected.Length -and
    (Get-KnownWrapperByteDigest $Content) -ceq (Get-KnownWrapperByteDigest $expected)
}

function Get-KnownWrapperEvidenceFiles($Wrapper) {
  $pattern = '^{0}\.emergency-containment-evidence-(?<digest>[0-9a-f]{{64}})$' -f
    [regex]::Escape([string]$Wrapper.Name)
  return @(Get-ChildItem -LiteralPath $Wrapper.DirectoryName -File -ErrorAction Stop |
    Where-Object { $_.Name -cmatch $pattern } |
    Sort-Object Name)
}

function Test-KnownDenyStubEvidence($Wrapper) {
  $evidenceFiles = @(Get-KnownWrapperEvidenceFiles $Wrapper)
  if ($evidenceFiles.Count -eq 0) { return $false }
  foreach ($evidenceFile in $evidenceFiles) {
    $expectedDigest = $evidenceFile.Name.Substring($evidenceFile.Name.Length - 64)
    $evidenceStream = $null
    try {
      $evidenceStream = Open-KnownExistingEvidenceExclusive $evidenceFile.FullName
      Assert-KnownEvidenceStreamDigest -Stream $evidenceStream -ExpectedDigest $expectedDigest
    } catch {
      return $false
    } finally {
      if ($null -ne $evidenceStream) { $evidenceStream.Dispose() }
    }
  }
  return $true
}

function Initialize-KnownEvidenceNativeApi {
  if ('ApplyPilot.KnownEvidenceNativeApi' -as [type]) { return }
  Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace ApplyPilot
{
    public static class KnownEvidenceNativeApi
    {
        private const uint GenericRead = 0x80000000;
        private const uint GenericWrite = 0x40000000;
        private const uint Delete = 0x00010000;
        private const uint CreateNew = 1;
        private const uint FileAttributeTemporary = 0x00000100;
        private const uint FileFlagDeleteOnClose = 0x04000000;

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateFileW(
            string fileName,
            uint desiredAccess,
            uint shareMode,
            IntPtr securityAttributes,
            uint creationDisposition,
            uint flagsAndAttributes,
            IntPtr templateFile);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetFileInformationByHandle(
            SafeFileHandle file,
            int informationClass,
            IntPtr information,
            uint bufferSize);

        [StructLayout(LayoutKind.Sequential)]
        private struct FileDispositionInfo
        {
            [MarshalAs(UnmanagedType.Bool)]
            public bool DeleteFile;
        }

        private static string ToExtendedLocalPath(string path)
        {
            if (String.IsNullOrWhiteSpace(path) ||
                path.StartsWith(@"\\", StringComparison.Ordinal) ||
                path.StartsWith(@"\\?\", StringComparison.Ordinal) ||
                path.StartsWith(@"\\.\", StringComparison.Ordinal))
                throw new IOException("evidence path is not a canonical local path");
            string fullPath = Path.GetFullPath(path);
            string root = Path.GetPathRoot(fullPath);
            if (String.IsNullOrEmpty(root) || root.Length != 3 ||
                root[1] != ':' || root[2] != '\\' || !Char.IsLetter(root[0]) ||
                fullPath.IndexOf(':', 2) >= 0)
                throw new IOException("evidence path is not a canonical local path");
            return @"\\?\" + fullPath;
        }

        private static void DeleteOpenedEvidence(SafeFileHandle handle)
        {
            var disposition = new FileDispositionInfo { DeleteFile = true };
            int size = Marshal.SizeOf<FileDispositionInfo>();
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try
            {
                Marshal.StructureToPtr(disposition, buffer, false);
                if (!SetFileInformationByHandle(handle, 4, buffer, (uint)size))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        public static SafeFileHandle CreateEvidenceStage(string path)
        {
            SafeFileHandle handle = CreateFileW(
                ToExtendedLocalPath(path),
                GenericRead | GenericWrite | Delete,
                0,
                IntPtr.Zero,
                CreateNew,
                FileAttributeTemporary | FileFlagDeleteOnClose,
                IntPtr.Zero);
            if (handle.IsInvalid)
            {
                int error = Marshal.GetLastWin32Error();
                handle.Dispose();
                throw new Win32Exception(error);
            }
            return handle;
        }

        public static FileStream PublishEvidence(string finalPath, byte[] content)
        {
            SafeFileHandle handle = null;
            FileStream stream = null;
            string extendedFinalPath = ToExtendedLocalPath(finalPath);
            try
            {
                handle = CreateFileW(
                    extendedFinalPath,
                    GenericRead | GenericWrite | Delete,
                    0,
                    IntPtr.Zero,
                    CreateNew,
                    0,
                    IntPtr.Zero);
                if (handle.IsInvalid)
                {
                    int error = Marshal.GetLastWin32Error();
                    handle.Dispose();
                    handle = null;
                    throw new Win32Exception(error);
                }
                stream = new FileStream(handle, FileAccess.ReadWrite);
                handle = null;
                stream.Write(content, 0, content.Length);
                stream.Flush(true);
                stream.Position = 0;
                var copied = new byte[content.Length];
                var offset = 0;
                while (offset < copied.Length)
                {
                    var read = stream.Read(copied, offset, copied.Length - offset);
                    if (read == 0) { break; }
                    offset += read;
                }
                if (stream.Length != content.Length || offset != content.Length ||
                    !System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(copied, content))
                {
                    throw new IOException("evidence publication verification failed");
                }
                stream.Position = 0;
                return stream;
            }
            catch
            {
                SafeFileHandle openedHandle = stream != null ? stream.SafeFileHandle : handle;
                try
                {
                    if (openedHandle != null && !openedHandle.IsInvalid && !openedHandle.IsClosed)
                        DeleteOpenedEvidence(openedHandle);
                }
                finally
                {
                    if (stream != null) { stream.Dispose(); }
                    else if (handle != null) { handle.Dispose(); }
                }
                throw;
            }
        }

        public static FileStream OpenExistingEvidence(string path)
        {
            return new FileStream(
                ToExtendedLocalPath(path),
                FileMode.Open,
                FileAccess.Read,
                FileShare.None);
        }
    }
}
'@
}

function New-KnownEvidenceStagePath([string]$FinalPath) {
  $directory = [IO.Path]::GetDirectoryName($FinalPath)
  $name = '.applypilot-emergency-containment-stage-{0}.tmp' -f [guid]::NewGuid().ToString('N')
  return [IO.Path]::Combine($directory, $name)
}

function Open-KnownEvidenceExclusive([string]$Path) {
  Initialize-KnownEvidenceNativeApi
  try {
    $stream = Open-KnownExistingEvidenceExclusive $Path
    return [pscustomobject]@{
      stream = $stream
      created = $false
      final_path = $Path
      stage_path = $null
    }
  } catch [IO.FileNotFoundException] {
  }

  $handle = $null
  try {
    $stagePath = $null
    for ($attempt = 0; $attempt -lt 16; $attempt++) {
      $stagePath = New-KnownEvidenceStagePath $Path
      try {
        $handle = [ApplyPilot.KnownEvidenceNativeApi]::CreateEvidenceStage($stagePath)
        break
      } catch [ComponentModel.Win32Exception] {
        if ($_.Exception.NativeErrorCode -notin @(80, 183) -or $attempt -eq 15) { throw }
      }
    }
    if ($null -eq $handle) { throw 'failed to create unique evidence staging file' }
    $stream = [IO.FileStream]::new($handle, [IO.FileAccess]::ReadWrite)
    $handle = $null
    return [pscustomobject]@{
      stream = $stream
      created = $true
      final_path = $Path
      stage_path = $stagePath
    }
  } catch {
    if ($null -ne $handle) { $handle.Dispose() }
    throw
  }
}

function Publish-NewKnownEvidenceByHandle($Lease) {
  Initialize-KnownEvidenceNativeApi
  if (-not $Lease.created -or [string]::IsNullOrWhiteSpace([string]$Lease.stage_path)) {
    throw 'only newly staged evidence can be published'
  }
  $stageStream = $Lease.stream
  $publishedStream = $null
  try {
    $content = Read-KnownWrapperStreamBytes $stageStream
    $publishedStream = [ApplyPilot.KnownEvidenceNativeApi]::PublishEvidence(
      [string]$Lease.final_path,
      $content
    )
  } finally {
    $stageStream.Dispose()
  }
  $Lease.stream = $publishedStream
  $Lease.stage_path = $null
  return $publishedStream
}

function Open-KnownExistingEvidenceExclusive([string]$Path) {
  Initialize-KnownEvidenceNativeApi
  return [ApplyPilot.KnownEvidenceNativeApi]::OpenExistingEvidence($Path)
}

function Write-KnownEvidenceBytes($Stream, [byte[]]$Content) {
  $Stream.Position = 0
  $Stream.SetLength(0)
  $Stream.Write($Content, 0, $Content.Length)
}

function Flush-KnownEvidenceStream($Stream) {
  $Stream.Flush($true)
}

function Assert-KnownEvidenceStreamDigest($Stream, [string]$ExpectedDigest) {
  $actualDigest = Get-KnownWrapperByteDigest (Read-KnownWrapperStreamBytes $Stream)
  if ($actualDigest -cne $ExpectedDigest) {
    throw 'wrapper evidence verification failed'
  }
}

function Get-KnownWrapperPaths {
  $roots = @(Get-KnownAuthorizedWrapperRoots)
  $paths = foreach ($root in $roots) {
    if ($root -and (Test-Path -LiteralPath $root -PathType Container)) {
      Get-ChildItem -LiteralPath $root -File -ErrorAction Stop |
        Where-Object { $script:GeneratedAcquisitionWrapperBasenames -contains $_.Name.ToLowerInvariant() }
    }
  }
  return @($paths | Sort-Object FullName -Unique)
}

function Get-TaskSnapshot {
  return @(Get-LegacyTasks | Sort-Object TaskPath, TaskName | ForEach-Object {
    $task = $_
    [ordered]@{
      name = $task.TaskName
      path = $task.TaskPath
      target_digest = Get-TextDigest ("{0}{1}" -f $task.TaskPath, $task.TaskName)
      state = [string]$task.State
      enabled = ([string]$task.State -ne 'Disabled')
      classification = if ($task.AuthorityClassification) {
        [string]$task.AuthorityClassification
      } else {
        'acquisition'
      }
      principal_digest = Get-TextDigest ([string]$task.Principal.UserId)
      actions = @($task.Actions | ForEach-Object {
        [ordered]@{
          executable = [IO.Path]::GetFileName([string]$_.Execute)
          arguments_digest = Get-TextDigest ([string]$_.Arguments)
          working_directory_digest = Get-TextDigest ([string]$_.WorkingDirectory)
        }
      })
    }
  })
}

function Get-ServiceSnapshot {
  return @(Get-LegacyServices | Sort-Object Name | ForEach-Object {
    [ordered]@{
      name = $_.Name
      target_digest = Get-TextDigest ([string]$_.Name)
      status = [string]$_.Status
      start_type = [string]$_.StartType
      classification = if ($_.AuthorityClassification) {
        [string]$_.AuthorityClassification
      } else {
        'acquisition'
      }
      executable_digest = Get-TextDigest ([string]$_.PathName)
    }
  })
}

function Get-WrapperSnapshot {
  return @(Get-KnownWrapperPaths | ForEach-Object {
    $wrapper = $_
    $lease = Open-KnownWrapperSnapshotExclusive ([string]$wrapper.FullName)
    $stream = $lease.Stream
    try {
      $bytes = Read-KnownWrapperStreamBytes $stream
      $lease.Validate()
      $content = ConvertFrom-KnownWrapperBytes $bytes
      $denyStub = Test-KnownWrapperDenyStub -Content $bytes -Extension $wrapper.Extension
      [ordered]@{
        name = $wrapper.Name
        path_digest = Get-TextDigest $wrapper.FullName
        identity_token = $lease.IdentityToken
        sha256 = Get-KnownWrapperByteDigest $bytes
        embedded_dsn = [bool](
          $content -match $script:SensitiveIdentifier -or $content -match $script:SensitiveUri
        )
        deny_stub = $denyStub
        evidence_verified = if ($denyStub) { Test-KnownDenyStubEvidence $wrapper } else { $null }
      }
    } finally {
      $lease.Dispose()
    }
  })
}

function Get-ProcessSnapshot {
  return @(Get-LegacyProcesses | Sort-Object ProcessId | ForEach-Object {
    [ordered]@{
      process_id = [int]$_.ProcessId
      name = [string]$_.Name
      target_digest = Get-ProcessStableIdentityDigest $_
      creation_digest = Get-TextDigest ([string]$_.CreationDate)
      executable_digest = Get-TextDigest ([string]$_.ExecutablePath)
      command_digest = Get-TextDigest ([string]$_.CommandLine)
      classification = if ($_.AuthorityClassification) {
        [string]$_.AuthorityClassification
      } else {
        'acquisition'
      }
    }
  })
}

function Get-CredentialReferenceSnapshot {
  return @($script:CredentialReferences | Sort-Object | ForEach-Object {
    [ordered]@{ name = $_; present = [bool][Environment]::GetEnvironmentVariable($_) }
  })
}

function Get-ControlEvidence {
  $fleet = [Environment]::GetEnvironmentVariable('FLEET_PG_DSN')
  $legacy = [Environment]::GetEnvironmentVariable('APPLYPILOT_FLEET_DSN')
  if ($fleet -and $legacy -and $fleet -cne $legacy) {
    return [ordered]@{
      admission_state = [ordered]@{ available = $false; authority_source = 'fleet_postgres'; error = 'inconsistent_dsn_references' }
      unresolved_attempt_counts = [ordered]@{ available = $false; authority_source = 'fleet_postgres'; error = 'inconsistent_dsn_references' }
    }
  }
  $dsn = if ($fleet) { $fleet } else { $legacy }
  $reference = if ($fleet) { 'FLEET_PG_DSN' } elseif ($legacy) { 'APPLYPILOT_FLEET_DSN' } else { $null }
  if (-not $dsn) {
    return [ordered]@{
      admission_state = [ordered]@{ available = $false; authority_source = 'fleet_postgres'; error = 'dsn_unavailable' }
      unresolved_attempt_counts = [ordered]@{ available = $false; authority_source = 'fleet_postgres'; error = 'dsn_unavailable' }
    }
  }
  $code = @'
import json, os
import psycopg
dsn = os.environ.get("FLEET_PG_DSN")
if not dsn:
    raise RuntimeError("fleet DSN unavailable")
with psycopg.connect(dsn) as conn:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("""SELECT paused, COALESCE(ats_paused,FALSE) AS ats_paused,
            ats_apply_mode, canary_enabled, canary_remaining,
            COALESCE(linkedin_apply_mode,'paused') AS linkedin_apply_mode,
            COALESCE(linkedin_canary_enabled,FALSE) AS linkedin_canary_enabled,
            linkedin_canary_remaining FROM fleet_config WHERE id=1""")
        state = cur.fetchone()
        counts = {}
        for table in ("apply_queue", "linkedin_queue"):
            cur.execute("SELECT to_regclass(%s) AS rel", (table,))
            if cur.fetchone()["rel"] is None:
                counts[table] = {"available": False}
                continue
            cur.execute(f"""SELECT
                count(*) FILTER (WHERE status='leased') AS leased,
                count(*) FILTER (WHERE COALESCE(apply_status,'') IN
                  ('crash_unconfirmed','no_confirmation','submission_uncertain','in_progress')) AS unresolved
                FROM {table}""")
            counts[table] = dict(cur.fetchone())
print(json.dumps({"state": state, "counts": counts}, default=str, separators=(",", ":")))
'@
  try {
    $raw = & py -3 -I -c $code 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'read-only control query failed' }
    $data = $raw | ConvertFrom-Json -AsHashtable
    return [ordered]@{
      admission_state = [ordered]@{
        available = [bool]$data.state
        authority_source = 'fleet_postgres'
        dsn_reference = $reference
        fields = $data.state
      }
      unresolved_attempt_counts = [ordered]@{
        available = $true
        authority_source = 'fleet_postgres'
        dsn_reference = $reference
        queues = $data.counts
      }
    }
  } catch {
    return [ordered]@{
      admission_state = [ordered]@{ available = $false; authority_source = 'fleet_postgres'; dsn_reference = $reference; error = 'query_unavailable' }
      unresolved_attempt_counts = [ordered]@{ available = $false; authority_source = 'fleet_postgres'; dsn_reference = $reference; error = 'query_unavailable' }
    }
  }
}

function Get-SupplementaryLocalAttemptCounts {
  $dbPath = [Environment]::GetEnvironmentVariable('APPLYPILOT_DB_PATH')
  if (-not $dbPath) { $dbPath = Join-Path $HOME '.applypilot\applypilot.db' }
  if (-not (Test-Path -LiteralPath $dbPath -PathType Leaf)) {
    return [ordered]@{ available = $false; authority_source = 'local_sqlite_supplementary' }
  }
  $code = @'
import json, sqlite3, sys
conn = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
out = {"available": True, "authority_source": "local_sqlite_supplementary", "ambiguous": 0, "in_progress": 0}
if "jobs" in tables:
    out["ambiguous"] = conn.execute("SELECT COUNT(*) FROM jobs WHERE apply_status IN ('crash_unconfirmed','no_confirmation','submission_uncertain')").fetchone()[0]
    out["in_progress"] = conn.execute("SELECT COUNT(*) FROM jobs WHERE apply_status='in_progress'").fetchone()[0]
print(json.dumps(out, separators=(",", ":")))
'@
  try {
    $raw = & py -3 -I -c $code $dbPath 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'read-only local query failed' }
    return ($raw | ConvertFrom-Json -AsHashtable)
  } catch {
    return [ordered]@{ available = $false; authority_source = 'local_sqlite_supplementary'; error = 'query_unavailable' }
  }
}

function Invoke-RejectionProbe {
  try {
    $command = (Get-Command applypilot -ErrorAction Stop).Source
    $lines = @(& $command apply --url 'https://invalid.example/emergency-probe' 2>&1 | ForEach-Object { [string]$_ })
    $exitCode = $LASTEXITCODE
    $text = $lines -join "`n"
    $verified = $exitCode -eq 78 -and $text -match '(?m)^APPLYPILOT_ADMISSION_DENIED:EMERGENCY_HOLD(?:\s|$)'
    if ($verified) {
      return [ordered]@{ status = 'verified'; verified = $true; decision = 'deny'; exit_code = $exitCode; output_digest = Get-TextDigest $text }
    }
    return [ordered]@{ status = 'unverified'; verified = $false; decision = $null; exit_code = $exitCode; output_digest = Get-TextDigest $text }
  } catch {
    return [ordered]@{ status = 'error'; verified = $false; decision = $null; exit_code = $null; error = 'console_probe_failed' }
  }
}

function Get-ContainmentSnapshot {
  $enumerationFailures = [Collections.Generic.List[object]]::new()
  $tasks = @()
  $services = @()
  $wrappers = @()
  $processes = @()
  try { $tasks = @(Get-TaskSnapshot) } catch {
    $enumerationFailures.Add([ordered]@{ source = 'scheduled_tasks'; error = 'enumeration_unavailable' })
  }
  try { $services = @(Get-ServiceSnapshot) } catch {
    $enumerationFailures.Add([ordered]@{ source = 'services'; error = 'enumeration_unavailable' })
  }
  try { $wrappers = @(Get-WrapperSnapshot) } catch {
    $enumerationFailures.Add([ordered]@{ source = 'wrappers'; error = 'enumeration_unavailable' })
  }
  try { $processes = @(Get-ProcessSnapshot) } catch {
    $enumerationFailures.Add([ordered]@{ source = 'processes'; error = 'enumeration_unavailable' })
  }
  $control = Get-ControlEvidence
  return [ordered]@{
    captured_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
    enumeration_failures = @($enumerationFailures)
    scheduled_tasks = $tasks
    services = $services
    wrapper_hashes = $wrappers
    process_identities = $processes
    credential_reference_names = @(Get-CredentialReferenceSnapshot)
    pause_admission_state = $control.admission_state
    unresolved_attempt_counts = $control.unresolved_attempt_counts
    supplementary_local_attempt_counts = Get-SupplementaryLocalAttemptCounts
  }
}

function Get-UnresolvedAfterState($Snapshot) {
  $unresolved = [Collections.Generic.List[object]]::new()
  foreach ($task in @($Snapshot.scheduled_tasks)) {
    if ([string]$task.classification -eq 'ambiguous') {
      $unresolved.Add([ordered]@{
        kind = 'task'
        target_digest = $task.target_digest
        conditions = @('ambiguous_command')
      })
    } elseif ([string]$task.state -ne 'Disabled') {
      $unresolved.Add([ordered]@{
        kind = 'task'
        target_digest = $task.target_digest
        conditions = @('not_disabled_or_still_runnable')
      })
    }
  }
  foreach ($service in @($Snapshot.services)) {
    if ([string]$service.classification -eq 'ambiguous') {
      $unresolved.Add([ordered]@{
        kind = 'service'
        target_digest = $service.target_digest
        conditions = @('ambiguous_command')
      })
      continue
    }
    $conditions = [Collections.Generic.List[string]]::new()
    if ([string]$service.status -ne 'Stopped') { $conditions.Add('not_stopped') }
    if ([string]$service.start_type -ne 'Disabled') { $conditions.Add('not_disabled') }
    if ($conditions.Count -gt 0) {
      $unresolved.Add([ordered]@{
        kind = 'service'
        target_digest = $service.target_digest
        conditions = @($conditions)
      })
    }
  }
  foreach ($process in @($Snapshot.process_identities)) {
    $conditions = if ([string]$process.classification -eq 'ambiguous') {
      'ambiguous_command'
    } else {
      'still_running'
    }
    $unresolved.Add([ordered]@{
      kind = 'process'
      target_digest = $process.target_digest
      conditions = @($conditions)
    })
  }
  foreach ($wrapper in @($Snapshot.wrapper_hashes)) {
    $conditions = [Collections.Generic.List[string]]::new()
    if ([bool]$wrapper.embedded_dsn) { $conditions.Add('embedded_dsn_present') }
    if (-not [bool]$wrapper.deny_stub) {
      $conditions.Add('exact_deny_stub_absent')
    } elseif (-not [bool]$wrapper.evidence_verified) {
      $conditions.Add('preserved_evidence_unverified')
    }
    if ($conditions.Count -gt 0) {
      $unresolved.Add([ordered]@{
        kind = 'wrapper'
        target_digest = $wrapper.path_digest
        conditions = @($conditions)
      })
    }
  }
  return @($unresolved)
}

if ($PSCmdlet.ParameterSetName -eq 'Evaluate') {
  try {
    $snapshot = $EvaluateAfterStateJson | ConvertFrom-Json -AsHashtable
    $unresolvedTargets = @(Get-UnresolvedAfterState $snapshot)
    [ordered]@{
      schema_version = 3
      mode = 'test'
      operation = 'evaluate_after_state'
      operational = $false
      non_operational_reasons = @('pure_data_evaluation')
      success = $false
      postconditions_satisfied = $unresolvedTargets.Count -eq 0
      unresolved_targets = $unresolvedTargets
      evidence_deleted = $false
    } | ConvertTo-Json -Depth 8 -Compress
  } catch {
    [ordered]@{
      schema_version = 3
      mode = 'test'
      operation = 'evaluate_after_state'
      operational = $false
      non_operational_reasons = @('pure_data_evaluation')
      success = $false
      postconditions_satisfied = $false
      unresolved_targets = @()
      rejection = 'invalid_snapshot_json'
      evidence_deleted = $false
    } | ConvertTo-Json -Compress
  }
  exit 2
}

function Invoke-RecordedAction([string]$Action, [string]$Target, [scriptblock]$Operation) {
  try {
    $null = & $Operation
  } catch {
    $script:Failures.Add([ordered]@{
      action = $Action
      target_digest = Get-TextDigest $Target
      error_type = $_.Exception.GetType().Name
    })
  }
}

function Disable-LegacyTasksAndServices {
  foreach ($task in @(Get-LegacyTasks)) {
    if ($task.AuthorityClassification -eq 'ambiguous') { continue }
    if ([string]$task.State -eq 'Running') {
      Invoke-RecordedAction 'stop_task' "$($task.TaskPath)$($task.TaskName)" { Stop-LegacyTask $task }
    }
    if ([string]$task.State -ne 'Disabled') {
      Invoke-RecordedAction 'disable_task' "$($task.TaskPath)$($task.TaskName)" { Disable-LegacyTask $task }
    }
  }
  foreach ($service in @(Get-LegacyServices)) {
    if ($service.AuthorityClassification -eq 'ambiguous') { continue }
    if ([string]$service.Status -ne 'Stopped') {
      Invoke-RecordedAction 'stop_service' $service.Name { Stop-LegacyService $service }
    }
    if ([string]$service.StartType -ne 'Disabled') {
      Invoke-RecordedAction 'disable_service' $service.Name { Disable-LegacyService $service }
    }
  }
}

function Stop-LegacyAcquisitionProcesses {
  foreach ($process in @(Get-LegacyProcesses)) {
    if ($process.AuthorityClassification -eq 'ambiguous') { continue }
    Invoke-RecordedAction 'stop_process' ([string]$process.ProcessId) { Stop-LegacyProcess $process }
  }
}

function Remove-EmbeddedWrapperDsns {
  param(
    [Parameter(Mandatory)]
    [AllowEmptyCollection()]
    [object[]]$ExpectedWrappers
  )

  $wrapperPaths = @(Get-KnownWrapperPaths)
  $expectedDigests = @($ExpectedWrappers | ForEach-Object { [string]$_.path_digest } | Sort-Object)
  $currentDigests = @($wrapperPaths | ForEach-Object { Get-TextDigest $_.FullName } | Sort-Object)
  if (($expectedDigests -join '|') -cne ($currentDigests -join '|')) {
    throw 'wrapper inventory changed after snapshot'
  }

  foreach ($wrapper in $wrapperPaths) {
    $pathDigest = Get-TextDigest $wrapper.FullName
    $expected = @($ExpectedWrappers | Where-Object { [string]$_.path_digest -ceq $pathDigest })
    Invoke-RecordedAction 'rewrite_wrapper' $wrapper.FullName {
      if ($expected.Count -ne 1 -or [string]::IsNullOrWhiteSpace(
          [string]$expected[0].identity_token)) {
        throw 'wrapper snapshot identity is unavailable'
      }
      $lease = Open-KnownWrapperMutationExclusive `
        -Path ([string]$wrapper.FullName) `
        -ExpectedIdentityToken ([string]$expected[0].identity_token)
      $stream = $lease.Stream
      $script:ActiveWrapperMutationLease = $lease
      $evidenceStream = $null
      try {
        $preimageBytes = Read-KnownWrapperStreamBytes $stream
        if (Test-KnownWrapperDenyStub -Content $preimageBytes -Extension $wrapper.Extension) {
          if (-not (Test-KnownDenyStubEvidence $wrapper)) {
            throw 'deny stub requires verified preserved evidence artifacts'
          }
          return
        }

        $preimageDigest = Get-KnownWrapperByteDigest $preimageBytes
        $evidencePath = "{0}.emergency-containment-evidence-{1}" -f $wrapper.FullName, $preimageDigest
        $evidenceLease = Open-KnownEvidenceExclusive $evidencePath
        $evidenceStream = $evidenceLease.stream
        if ($evidenceLease.created) {
          Write-KnownEvidenceBytes -Stream $evidenceStream -Content $preimageBytes
          Flush-KnownEvidenceStream $evidenceStream
          Assert-KnownEvidenceStreamDigest -Stream $evidenceStream -ExpectedDigest $preimageDigest
          $evidenceStream = Publish-NewKnownEvidenceByHandle $evidenceLease
        } else {
          Assert-KnownEvidenceStreamDigest -Stream $evidenceStream -ExpectedDigest $preimageDigest
        }

        $denyBytes = Get-KnownWrapperDenyStubBytes $wrapper.Extension
        try {
          Write-KnownWrapperDenyStub -Stream $stream -Content $denyBytes
          $writtenBytes = Read-KnownWrapperStreamBytes $stream
          if (-not (Test-KnownWrapperDenyStub -Content $writtenBytes -Extension $wrapper.Extension)) {
            throw 'wrapper deny stub verification failed'
          }
        } catch {
          $writeFailure = $_
          try {
            Restore-KnownWrapperBytes -Stream $stream -Content $preimageBytes
          } catch {
            throw 'wrapper deny stub write failed and original restoration failed'
          }
          throw $writeFailure
        }
        Assert-KnownEvidenceStreamDigest -Stream $evidenceStream -ExpectedDigest $preimageDigest
      } finally {
        $script:ActiveWrapperMutationLease = $null
        if ($null -ne $evidenceStream) { $evidenceStream.Dispose() }
        $lease.Dispose()
      }
    }
  }
}

function Invoke-ContainmentOrchestration {
  param(
    [Parameter(Mandatory)]
    [ValidateSet('inspect', 'contain')]
    [string]$Operation
  )

  $script:Failures.Clear()
  $script:SkippedActions.Clear()
  $before = Get-ContainmentSnapshot
  $beforeProbe = Invoke-RejectionProbe
  if ($Operation -eq 'contain' -and @($before.enumeration_failures).Count -eq 0) {
    try {
      Disable-LegacyTasksAndServices
      Stop-LegacyAcquisitionProcesses
      Remove-EmbeddedWrapperDsns -ExpectedWrappers @($before.wrapper_hashes)
    } catch {
      $script:Failures.Add([ordered]@{
        action = 'containment_actions'
        target_digest = $null
        error_type = 'enumeration_unavailable'
      })
    }
  }
  $after = Get-ContainmentSnapshot
  $afterProbe = Invoke-RejectionProbe
  $unresolvedTargets = @(Get-UnresolvedAfterState $after)
  $enumerationFailures = @(
    @($before.enumeration_failures) + @($after.enumeration_failures) |
      Sort-Object source, error -Unique
  )

  return [ordered]@{
    operation = $Operation
    postconditions_satisfied = $enumerationFailures.Count -eq 0 -and $unresolvedTargets.Count -eq 0
    enumeration_failures = $enumerationFailures
    unresolved_targets = $unresolvedTargets
    before = $before
    before_rejection_probe = $beforeProbe
    after = $after
    after_rejection_probe = $afterProbe
    rejection_probe_satisfied = $afterProbe.verified -and $afterProbe.decision -eq 'deny'
    failures = @($script:Failures)
    skipped_actions = @($script:SkippedActions)
    evidence_deleted = $false
  }
}

function Get-ContainmentDisposition($Core) {
  $success = if ([string]$Core.operation -eq 'contain') {
    @($Core.failures).Count -eq 0 -and
      [bool]$Core.postconditions_satisfied -and
      [bool]$Core.rejection_probe_satisfied
  } else {
    [bool]$Core.postconditions_satisfied -and [bool]$Core.rejection_probe_satisfied
  }
  return [ordered]@{
    success = $success
    exit_code = if ($success) { 0 } else { 1 }
  }
}

if ($PSCmdlet.ParameterSetName -eq 'DefinitionImport') {
  if ($MyInvocation.InvocationName -eq '.') { return }
  [ordered]@{
    schema_version = 3
    mode = 'test'
    operation = 'definition_import'
    operational = $false
    success = $false
    rejection = 'definition_import_requires_dot_source'
    evidence_deleted = $false
  } | ConvertTo-Json -Compress
  exit 2
}

$operation = if ($Contain) { 'contain' } else { 'inspect' }
$core = Invoke-ContainmentOrchestration -Operation $operation
$disposition = Get-ContainmentDisposition -Core $core

[ordered]@{
  schema_version = 3
  mode = 'operational'
  operation = $core.operation
  operational = $true
  non_operational_reasons = @()
  success = $disposition.success
  postconditions_satisfied = $core.postconditions_satisfied
  enumeration_failures = $core.enumeration_failures
  unresolved_targets = $core.unresolved_targets
  before = $core.before
  before_rejection_probe = $core.before_rejection_probe
  after = $core.after
  after_rejection_probe = $core.after_rejection_probe
  failures = $core.failures
  skipped_actions = $core.skipped_actions
  evidence_deleted = $core.evidence_deleted
} | ConvertTo-Json -Depth 12 -Compress

if ($disposition.exit_code -ne 0) { exit $disposition.exit_code }
