let DATA = null;
let CURRENT_TIER_FILTER = 0;
let CURRENT_VIEW = 'overview';

document.addEventListener('DOMContentLoaded', async () => {
  DATA = await PI.loadAll();
  if (!DATA.raw.roles) {
    document.querySelector('main').innerHTML = '<div class="loading">Failed to load data from <code>.project-intelligence/</code>. Launch with <code>python -m http.server</code>.</div>';
    return;
  }
  setupTabs();
  setupTierFilter();
  renderOverview();
  renderRoles();
  renderJourneys();
  renderFeatures();
  renderCapabilities();
  renderCode();
  renderRisks();
  showView('overview');
});

function setupTabs() {
  document.querySelectorAll('nav.tabs button').forEach(btn => {
    btn.addEventListener('click', () => showView(btn.dataset.tab));
  });
}

function showView(name) {
  CURRENT_VIEW = name;
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('nav.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + name).classList.add('active');
  const active = document.querySelector(`nav.tabs button[data-tab="${name}"]`);
  if (active) active.classList.add('active');
  window.scrollTo(0, 0);
}

function setupTierFilter() {
  document.querySelectorAll('.tier-filter button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tier-filter button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      CURRENT_TIER_FILTER = parseInt(btn.dataset.tier) || 0;
      renderRoles();
      renderJourneys();
    });
  });
}

function badgeTier(tier) {
  return `<span class="badge badge-tier-${tier}">Tier ${tier}</span>`;
}
function badgeStatus(status) {
  if (status === 'confirmed') return `<span class="badge badge-confirmed">✓ ${status}</span>`;
  if (status === 'inferred') return `<span class="badge badge-inferred">${status}</span>`;
  return `<span class="badge badge-needs_review">${status || 'needs_review'}</span>`;
}

function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function tierLabel(t) {
  return { 1: 'Discovery', 2: 'AI Scoring & Tailoring', 3: 'Full Auto-Apply' }[t] || 'Unknown';
}

// ============ OVERVIEW ============
function renderOverview() {
  const root = document.getElementById('view-overview');
  const cto = DATA.raw.cto || {};
  const q = cto.quantitative_landscape || {};

  const roleCount = DATA.raw.roles.roles.length;
  const journeyCount = DATA.raw.journeys.journeys.length;
  const stepCount = DATA.raw.journeys.journeys.reduce((s,j) => s + (j.steps?.length || 0), 0);
  const featureCount = DATA.raw.features.features.length;
  const capCount = DATA.raw.caps.capabilities.length;

  root.innerHTML = `
    <h1>ApplyPilot — Product Intelligence</h1>
    <p class="subtitle">Reverse-engineered from source code. All conclusions have direct evidence.</p>

    <div class="stats-row">
      <div class="stat-pill"><div class="val">${roleCount}</div><div class="lbl">Human Roles</div></div>
      <div class="stat-pill"><div class="val">${journeyCount}</div><div class="lbl">User Journeys</div></div>
      <div class="stat-pill"><div class="val">${stepCount}</div><div class="lbl">User Actions</div></div>
      <div class="stat-pill"><div class="val">${featureCount}</div><div class="lbl">Business Features</div></div>
      <div class="stat-pill"><div class="val">${capCount}</div><div class="lbl">Atomic Capabilities</div></div>
      <div class="stat-pill"><div class="val">${DATA.raw.inventory?.symbols?.symbols?.length || '?'}</div><div class="lbl">Source Symbols</div></div>
    </div>

    <div class="section-title">Product Architecture</div>
    <div class="arch-layer">
      <h3>Presentation Layer</h3>
      <div class="arch-comps">
        <span class="arch-comp">cli.py (6 commands)</span>
        <span class="arch-comp">wizard/init.py (5-step interactive)</span>
        <span class="arch-comp">status (rich tables)</span>
        <span class="arch-comp">dashboard (HTML)</span>
      </div>
    </div>
    <div class="arch-layer">
      <h3>Orchestration Layer</h3>
      <div class="arch-comps">
        <span class="arch-comp">pipeline.py (6-stage orchestrator)</span>
        <span class="arch-comp">apply/launcher.py (worker loop + MCP)</span>
        <span class="arch-comp">config.py (Tier gate + dependency detection)</span>
      </div>
    </div>
    <div class="arch-layer">
      <h3>Domain Engines</h3>
      <div class="arch-comps">
        <span class="arch-comp">discovery/ (JobSpy + Workday + SmartExtract)</span>
        <span class="arch-comp">enrichment/ (3-tier cascade)</span>
        <span class="arch-comp">scoring/ (scorer + tailor + cover + validator)</span>
        <span class="arch-comp">apply/ (Chrome + Claude Agent)</span>
      </div>
    </div>
    <div class="arch-layer">
      <h3>Infrastructure Layer</h3>
      <div class="arch-comps">
        <span class="arch-comp">database.py (SQLite singleton)</span>
        <span class="arch-comp">llm.py (multi-provider)</span>
        <span class="arch-comp">config.py (Paths + AppConfig)</span>
      </div>
    </div>

    <div class="section-title">Tier Capability Boundaries</div>
    ${renderTierLayer()}

    <div class="section-title">How it works: the SQLite Conveyor Belt</div>
    <p style="color:var(--text);font-size:13px;margin-bottom:12px;">
      The smartest design decision: <strong>SQLite jobs.db as the message queue</strong>. Each stage independently polls for pending records via its own PENDING_SQL.
      No explicit message broker, no stage coupling.
    </p>
    <pre>discover ──upsert url/title/company──► jobs.db
    │ [WHERE detail_scraped_at IS NULL]
    ▼
enrich ──── write full_description ──► jobs.db
    │ [WHERE full_description IS NOT NULL AND fit_score IS NULL]
    ▼
score ───── write fit_score (0-10) ──► jobs.db
    │ [WHERE fit_score >= min_score AND tailored_resume_path IS NULL]
    ▼
tailor ──── write tailored_resume_path ─► jobs.db
    │ [WHERE tailored_resume_path IS NOT NULL AND cover_letter_path IS NULL]
    ▼
cover ───── write cover_letter_path ──► jobs.db
    │ [WHERE tailored_resume_path LIKE '%.yaml']
    ▼
pdf ──────── render .yaml → .pdf ──────► jobs.db</pre>
  `;
}

function renderTierLayer() {
  const tiers = [
    { n: 1, title: 'Discovery', cond: 'Python + pip only', cmds: ['init','doctor','run discover','run enrich','status','dashboard','apply --mark-applied/failed/reset'] },
    { n: 2, title: 'AI Scoring & Tailoring', cond: 'Tier 1 + LLM API key env var', cmds: ['run score','run tailor','run cover','run pdf','run all','run --stream'] },
    { n: 3, title: 'Full Auto-Apply', cond: 'Tier 2 + claude CLI + Chrome + Node.js', cmds: ['apply','apply --gen --url','apply --workers N','apply --continuous'] },
  ];
  return tiers.map(t => `
    <div style="border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:10px;border-left:3px solid var(--tier${t.n});">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <h3 style="margin:0;"><span class="badge badge-tier-${t.n}">Tier ${t.n}</span> ${t.title}</h3>
        <code style="font-size:12px;">config.py:get_tier() → return ${t.n}</code>
      </div>
      <p style="color:var(--text-dim);font-size:12px;margin-bottom:8px;"><strong>Unlock:</strong> ${esc(t.cond)}</p>
      <div style="display:flex;flex-wrap:wrap;gap:4px;">${t.cmds.map(c => `<code style="font-size:11px;background:var(--bg);color:var(--text);padding:2px 6px;border-radius:3px;">${esc(c)}</code>`).join('')}</div>
    </div>
  `).join('');
}

// ============ ROLES ============
function renderRoles() {
  const root = document.getElementById('view-roles');
  const roles = DATA.raw.roles.roles;

  root.innerHTML = `
    <h1>Human Roles</h1>
    <p class="subtitle">Products from the user's perspective. Click a role to see its journeys.</p>
    <div class="grid-cards">
      ${roles.filter(r => CURRENT_TIER_FILTER === 0 || Object.keys(r.accessible_commands || {}).some(k => parseInt(k.split('_').pop()) >= CURRENT_TIER_FILTER)).map(r => {
        const tiersCovered = new Set();
        Object.keys(r.accessible_commands || {}).forEach(k => {
          const m = k.match(/tier[_\-]?(\d)/i);
          if (m) tiersCovered.add(parseInt(m[1]));
        });
        const maxTier = tiersCovered.size ? Math.max(...tiersCovered) : 1;
        return `
          <div class="card" onclick="showRole('${r.role_id}')">
            <div class="card-header">
              <div class="title">${esc(r.human_readable || r.role_name)}</div>
              ${badgeTier(maxTier)}
            </div>
            <div class="meta">${esc(r.role_id)} · ${(r.associated_journeys || []).length} journeys</div>
            <div class="desc">${esc(r.description)}</div>
          </div>
        `;
      }).join('')}
    </div>
    <p style="color:var(--text-dim);font-size:12px;margin-top:20px;">
      Note: ApplyPilot has only one human role. Tier 1/2/3 is a <strong>capability detection</strong> mechanism (install dependencies to unlock more commands),
      not role-based access control. The same job_seeker can execute all tiers.
    </p>
  `;
}

function showRole(roleId) {
  const root = document.getElementById('view-roles');
  const role = DATA.raw.roles.roles.find(r => r.role_id === roleId);
  if (!role) return;
  const associatedJourneys = (role.associated_journeys || []).map(id => DATA.J[id]).filter(Boolean);

  root.innerHTML = `
    <button class="back-btn" onclick="renderRoles()">← Back to Roles</button>
    <h1>${esc(role.human_readable || role.role_name)}</h1>
    <p class="subtitle">${esc(role.description)}</p>

    <div class="section-title">Accessible Commands by Tier</div>
    <table>
      <thead><tr><th>Tier</th><th>Commands</th></tr></thead>
      <tbody>
        ${Object.entries(role.accessible_commands || {}).map(([k, v]) => {
          const m = k.match(/(\d)/);
          const t = m ? parseInt(m[1]) : 1;
          return `<tr><td>${badgeTier(t)} ${t === 1 ? 'Base' : 'Unlock'}</td><td><code style="font-size:11px;">${v.map(esc).join('</code> · <code style="font-size:11px;">')}</code></td></tr>`;
        }).join('')}
      </tbody>
    </table>

    <div class="section-title">Associated Journeys</div>
    <div class="grid-cards">
      ${associatedJourneys.map(j => `
        <div class="card" onclick="showJourney('${j.journey_id}')">
          <div class="card-header">
            <div class="title">${esc(j.name)}</div>
            ${badgeTier(j.tier_required || 1)}
          </div>
          <div class="meta">${esc(j.journey_id)} · ${(j.steps?.length || 0)} steps</div>
          <div class="desc">${esc(j.description)}</div>
        </div>
      `).join('')}
    </div>

    <div class="section-title">Evidence</div>
    <div class="evidence-list">
      ${(role.evidence || []).map(e => `<li>${esc(e.file || '')}${e.line_range ? ':' + esc(e.line_range) : ''} — ${esc(e.detail || e.description || '')}</li>`).join('')}
    </div>
  `;
}

// ============ JOURNEYS ============
function renderJourneys() {
  const root = document.getElementById('view-journeys');
  const journeys = DATA.raw.journeys.journeys.filter(j => CURRENT_TIER_FILTER === 0 || (j.tier_required || 1) === CURRENT_TIER_FILTER);

  root.innerHTML = `
    <h1>User Journeys</h1>
    <p class="subtitle">Click a journey to see its ordered steps with code entry points.</p>
    <div class="timeline">
      ${journeys.map(j => `
        <div class="timeline-item tier-${j.tier_required || 1}">
          <div class="step-block" style="cursor:pointer;" onclick="showJourney('${j.journey_id}')">
            <div class="step-title">
              ${badgeTier(j.tier_required || 1)}
              ${esc(j.name)}
              ${badgeStatus(j.status)}
            </div>
            <div class="step-desc">${esc(j.description)}</div>
            <div class="step-meta">
              <span class="badge badge-feature">${esc(j.cli_entry || '')}</span>
              <span class="badge badge-system">${(j.steps?.length || 0)} steps</span>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function showJourney(journeyId) {
  const root = document.getElementById('view-journeys');
  const j = DATA.J[journeyId];
  if (!j) return;

  const steps = j.steps || [];

  root.innerHTML = `
    <button class="back-btn" onclick="renderJourneys()">← Back to Journeys</button>
    <h1>${esc(j.name)}</h1>
    <p class="subtitle">
      ${badgeTier(j.tier_required || 1)}
      ${badgeStatus(j.status)}
      ${(j.feature_ids || []).map(fid => `<span class="badge badge-feature">${esc(fid)}</span>`).join(' ')}
    </p>
    <p style="color:var(--text);font-size:13px;margin-bottom:20px;">${esc(j.description)}</p>

    <div class="section-title">Evidence</div>
    <div class="evidence-list">
      ${(j.evidence || []).map(e => {
        const parsed = PI.parseEvidenceString(e);
        if (parsed && parsed.file) {
          return `<li><span class="ev-file">${esc(parsed.file)}</span><span class="ev-line">:</span><span class="ev-line">${esc(parsed.lines)}</span><span class="ev-desc">${esc(parsed.desc)}</span></li>`;
        }
        return `<li>${esc(e)}</li>`;
      }).join('')}
    </div>

    <div class="section-title">User Actions (${steps.length} ordered steps)</div>
    <div class="timeline">
      ${steps.map((s, idx) => renderStepBlock(s, idx + 1, j)).join('')}
    </div>
  `;
}

function renderStepBlock(step, orderNum, journey) {
  const features = PI.featuresForStep(step, DATA.FE);
  const caps = PI.capabilitiesForStep(step, DATA.FE, DATA.C);
  const parsedTrigger = PI.parseTriggerString(step.trigger);
  const stepTypeClass = step.step_type === 'user_action' ? 'user_action' : (step.step_type === 'system_action' ? 'system_action' : (step.step_type === 'system_output' ? 'system_output' : 'user_interaction'));
  const evidenceParsed = (step.evidence || []).map(e => PI.parseEvidenceString(e));

  const cliEntry = findCLIEntry(step, journey);
  const backendEntry = findBackendEntry(step, parsedTrigger);

  return `
    <div class="timeline-item tier-${journey.tier_required || 1} step-confirmed">
      <div class="step-block ${stepTypeClass}">
        <div class="step-title">
          <span style="color:var(--text-dim);font-size:11px;">#${orderNum}</span>
          ${esc(step.user_action || step.step_id)}
          ${badgeStatus(step.status)}
          ${step.is_system_behaviour ? '<span class="badge badge-system">system</span>' : ''}
          ${step.is_optional ? '<span class="badge badge-system">optional</span>' : ''}
        </div>
        <div class="step-desc">${esc(step.description)}</div>

        ${cliEntry ? `
          <div class="entry-block">
            <div class="entry-label">CLI / Frontend Entry</div>
            <div class="entry-main">${esc(cliEntry.command)}</div>
            <div class="entry-sub">${esc(cliEntry.source)}</div>
          </div>
        ` : ''}

        ${backendEntry ? `
          <div class="entry-block">
            <div class="entry-label">Backend / Core Implementation</div>
            <div class="entry-main">${esc(backendEntry.main)}</div>
            ${backendEntry.chain ? backendEntry.chain.map(c => `<div class="entry-sub">→ ${esc(c)}</div>`).join('') : ''}
            ${backendEntry.file ? `<div class="entry-sub">file: ${esc(backendEntry.file)}</div>` : ''}
            ${backendEntry.symbol ? `<div class="entry-sub">symbol: <code>${esc(backendEntry.symbol)}</code></div>` : ''}
          </div>
        ` : ''}

        ${features.length ? `
          <div class="step-meta">
            <span class="badge badge-feature">Feature: ${esc(features[0].name)} (${esc(features[0].feature_id)})</span>
          </div>
        ` : ''}

        ${caps.length ? `
          <details style="margin-bottom:8px;">
            <summary>Related Atomic Capabilities (${caps.length})</summary>
            <div style="padding:8px 0;">
              ${caps.map(c => `
                <details style="margin-bottom:4px;background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:0;">
                  <summary style="padding:6px 10px;font-size:12px;">
                    <code style="color:var(--accent);">${esc(c.capability_id)}</code> ${esc(c.name)}
                    ${badgeStatus(c.status)}
                  </summary>
                  <div style="padding:0 10px 10px;font-size:12px;color:var(--text);">
                    <p style="margin-bottom:6px;">${esc(c.description)}</p>
                    ${(c.source_symbols || []).length ? `
                      <div style="color:var(--text-dim);font-size:11px;">Source:</div>
                      ${(c.source_symbols || []).map(sid => {
                        const sym = DATA.S[sid];
                        const file = PI.resolveFileFromSymbol(sym, DATA.F);
                        if (!sym) return '';
                        return `<div style="font-family:var(--font-mono);font-size:11px;color:var(--text);">
                          ${esc(sym.kind)} <code>${esc(sym.qualified_name)}</code> line ${sym.line_start}
                          <span style="color:var(--text-dim);">— ${esc(file)}</span>
                        </div>`;
                      }).join('')}
                    ` : ''}
                  </div>
                </details>
              `).join('')}
            </div>
          </details>
        ` : ''}

        ${(evidenceParsed || []).length ? `
          <details style="margin-bottom:8px;">
            <summary>Evidence (${evidenceParsed.length})</summary>
            <ul class="evidence-list" style="margin-top:8px;">
              ${evidenceParsed.filter(Boolean).map(e => `<li><span class="ev-file">${esc(e.file)}</span><span class="ev-line">:</span><span class="ev-line">${esc(e.lines)}</span> — <span class="ev-desc">${esc(e.desc)}</span></li>`).join('')}
            </ul>
          </details>
        ` : ''}

        ${step.preconditions?.length ? `<div style="color:var(--text-dim);font-size:11px;margin-top:6px;"><strong>Preconditions:</strong> ${step.preconditions.map(esc).join(', ')}</div>` : ''}
        ${step.outcome ? `<div style="color:var(--text-dim);font-size:11px;"><strong>Outcome:</strong> ${esc(step.outcome)}</div>` : ''}
      </div>
    </div>
  `;
}

function findCLIEntry(step, journey) {
  if (journey.cli_entry) {
    return { command: journey.cli_entry, source: step.trigger || 'cli.py' };
  }
  const trigger = step.trigger || '';
  const cliMatch = trigger.match(/cli\.py[:：]([\w]+)/);
  if (cliMatch) return { command: `applypilot ${cliMatch[1]}`, source: trigger };
  return null;
}

function findBackendEntry(step, parsedTrigger) {
  const trigger = step.trigger || '';
  if (!parsedTrigger.primary && !parsedTrigger.chain.length && !trigger) return null;
  const fileMatch = trigger.match(/([\w\/]+\.py)[:：]([\w\.]+)/);
  return {
    main: parsedTrigger.primary || trigger,
    chain: parsedTrigger.chain,
    file: fileMatch ? fileMatch[1] : '',
    symbol: fileMatch ? fileMatch[2] : '',
  };
}

// ============ FEATURES ============
function renderFeatures() {
  const root = document.getElementById('view-features');
  const features = DATA.raw.features.features;

  root.innerHTML = `
    <h1>Business Features</h1>
    <p class="subtitle">10 features · 100% covered by user journeys. Click to see capabilities and code.</p>
    <div class="grid-cards">
      ${features.map(f => `
        <div class="card" onclick="showFeature('${f.feature_id}')">
          <div class="card-header">
            <div class="title">${esc(f.name)}</div>
            ${badgeStatus(f.status)}
          </div>
          <div class="meta">${esc(f.feature_id)} · ${(f.capability_ids || []).length} capabilities</div>
          <div class="desc">${esc(f.description)}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function showFeature(fid) {
  const root = document.getElementById('view-features');
  const f = DATA.FE[fid];
  if (!f) return;

  const caps = (f.capability_ids || []).map(cid => DATA.C[cid]).filter(Boolean);
  const symbols = PI.findSymbolsForFeature(f, DATA.S, DATA.F);
  const journeys = DATA.raw.journeys.journeys.filter(j => (j.feature_ids || []).includes(fid));

  root.innerHTML = `
    <button class="back-btn" onclick="renderFeatures()">← Back to Features</button>
    <h1>${esc(f.name)}</h1>
    <p class="subtitle">${esc(f.feature_id)} · ${badgeStatus(f.status)} · ${esc(f.feature_type || '')}</p>
    <p style="margin-bottom:16px;">${esc(f.description)}</p>

    ${f.cli_entry ? `<div class="entry-block"><div class="entry-label">CLI Entry</div><div class="entry-main">${esc(f.cli_entry)}</div></div>` : ''}

    <div class="section-title">Code Entry Candidates</div>
    ${symbols.length ? `
      <div class="grid-cards">
        ${symbols.map(s => `
          <div class="card" style="cursor:default;">
            <div class="title" style="font-family:var(--font-mono);font-size:12px;">${esc(s.qualified_name)}</div>
            <div class="meta">${esc(s.file)} · line ${s.line}</div>
          </div>
        `).join('')}
      </div>
    ` : (f.entry_candidates || []).length ? `<pre>${(f.entry_candidates || []).map(esc).join('\n')}</pre>` : '<div class="missing">No direct symbol mapping found.</div>'}

    <div class="section-title">Related Journeys</div>
    ${journeys.length ? `
      <div class="grid-cards">
        ${journeys.map(j => `
          <div class="card" onclick="showJourney('${j.journey_id}')">
            <div class="card-header">
              <div class="title">${esc(j.name)}</div>
              ${badgeTier(j.tier_required || 1)}
            </div>
            <div class="meta">${esc(j.journey_id)}</div>
          </div>
        `).join('')}
      </div>
    ` : '<div class="missing">No journeys reference this feature.</div>'}

    <div class="section-title">Atomic Capabilities (${caps.length})</div>
    ${caps.map(c => `
      <details style="margin-bottom:6px;background:var(--bg-card);border:1px solid var(--border);border-radius:6px;">
        <summary>
          <span class="badge badge-cap">${esc(c.capability_id)}</span>
          ${esc(c.name)} ${badgeStatus(c.status)}
        </summary>
        <div style="padding:10px 0;font-size:13px;color:var(--text);">
          ${esc(c.description)}
          ${(c.source_symbols || []).length ? `
            <div style="margin-top:8px;">
              <div style="color:var(--text-dim);font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Source Symbols</div>
              ${c.source_symbols.map(sid => {
                const sym = DATA.S[sid];
                const file = PI.resolveFileFromSymbol(sym, DATA.F);
                if (!sym) return '';
                return `<div style="font-family:var(--font-mono);font-size:12px;margin-bottom:3px;">
                  <code>${esc(sym.qualified_name)}</code> <span style="color:var(--text-dim);">line ${sym.line_start}</span>
                  <span style="color:var(--text-dim);">— ${esc(file)}</span>
                </div>`;
              }).join('')}
            </div>
          ` : ''}
        </div>
      </details>
    `).join('')}
  `;
}

// ============ CAPABILITIES ============
function renderCapabilities() {
  const root = document.getElementById('view-capabilities');
  const caps = DATA.raw.caps.capabilities;

  root.innerHTML = `
    <h1>Atomic Capabilities</h1>
    <p class="subtitle">28 independently-callable technical/business capabilities. Each traced to source symbols.</p>
    <table>
      <thead><tr><th>ID</th><th>Name</th><th>Type</th><th>Status</th><th>Source Symbols</th></tr></thead>
      <tbody>
        ${caps.map(c => `
          <tr style="cursor:pointer;" onclick="showCapability('${c.capability_id}')">
            <td><code>${esc(c.capability_id)}</code></td>
            <td>${esc(c.name)}</td>
            <td><span class="badge badge-system">${esc(c.capability_type)}</span></td>
            <td>${badgeStatus(c.status)}</td>
            <td>${(c.source_symbols || []).length}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

function showCapability(cid) {
  const root = document.getElementById('view-capabilities');
  const c = DATA.C[cid];
  if (!c) return;
  const syms = PI.findSymbolsForCapability(c, DATA.S, DATA.F);
  const features = (DATA.raw.features.features || []).filter(f => (f.capability_ids || []).includes(cid));

  root.innerHTML = `
    <button class="back-btn" onclick="renderCapabilities()">← Back to Capabilities</button>
    <h1><code style="color:var(--accent);">${esc(c.capability_id)}</code> ${esc(c.name)}</h1>
    <p class="subtitle">${esc(c.capability_type)} · ${badgeStatus(c.status)}</p>
    <p style="margin-bottom:16px;">${esc(c.description)}</p>

    <div class="section-title">Source Symbols (${syms.length})</div>
    ${syms.length ? `
      <table>
        <thead><tr><th>Kind</th><th>Qualified Name</th><th>Line</th><th>File</th></tr></thead>
        <tbody>
          ${syms.map(s => `
            <tr>
              <td><span class="badge badge-system">${esc(s.kind)}</span></td>
              <td><code>${esc(s.qualified_name)}</code></td>
              <td>${s.line}</td>
              <td style="font-family:var(--font-mono);font-size:12px;">${esc(s.file)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    ` : '<div class="missing">No source symbols mapped for this capability.</div>'}

    <div class="section-title">Features Using This Capability</div>
    ${features.length ? features.map(f => `<div class="badge badge-feature" style="margin:2px;">${esc(f.feature_id)}: ${esc(f.name)}</div>`).join(' ') : '<div class="missing">No features reference this capability.</div>'}

    ${c.inputs?.length ? `<div class="section-title">Inputs</div><ul style="padding-left:20px;"><li>${c.inputs.map(esc).join('</li><li>')}</li></ul>` : ''}
    ${c.outputs?.length ? `<div class="section-title">Outputs</div><ul style="padding-left:20px;"><li>${c.outputs.map(esc).join('</li><li>')}</li></ul>` : ''}
    ${c.side_effects?.length ? `<div class="section-title">Side Effects</div><ul style="padding-left:20px;"><li>${c.side_effects.map(esc).join('</li><li>')}</li></ul>` : ''}
    ${c.dependencies?.length ? `<div class="section-title">Dependencies</div>${c.dependencies.map(d => `<code style="font-size:11px;margin-right:4px;">${esc(d)}</code>`).join('')}` : ''}
    ${c.open_questions?.length ? `<div class="section-title">Open Questions</div>${c.open_questions.map(q => `<div style="color:var(--accent-yellow);font-size:12px;margin-bottom:4px;">❓ ${esc(q)}</div>`).join('')}` : ''}
  `;
}

// ============ CODE ============
function renderCode() {
  const root = document.getElementById('view-code');
  const cto = DATA.raw.cto || {};
  const arch = cto.product_architecture || {};
  const mlm = (cto.module_responsibility_matrix || {});

  const symbolsByFile = {};
  (DATA.raw.symbols?.symbols || []).forEach(s => {
    const f = PI.resolveFileFromSymbol(s, DATA.F);
    if (!symbolsByFile[f]) symbolsByFile[f] = [];
    symbolsByFile[f].push(s);
  });

  const fileList = Object.entries(symbolsByFile).sort((a,b) => a[0].localeCompare(b[0]));

  root.innerHTML = `
    <h1>Technical Architecture</h1>
    <p class="subtitle">Module responsibilities, data flow, and source code browser.</p>

    <div class="section-title">Data Flow: The SQLite Conveyor Belt</div>
    ${renderDataFlow()}

    <div class="section-title">Module Responsibility Matrix</div>
    <table>
      <thead><tr><th>Module</th><th>Responsibility</th><th>Tier Access</th></tr></thead>
      <tbody>
        ${Object.entries(mlm).map(([k,v]) => `
          <tr>
            <td style="font-family:var(--font-mono);font-size:12px;">${esc(k)}</td>
            <td>${esc(v.responsibility || '')}</td>
            <td>${esc(v.tier_access || '')}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>

    <div class="section-title">Source Files & Symbols</div>
    ${fileList.map(([file, syms]) => `
      <details>
        <summary>
          <code>${esc(file)}</code>
          <span style="color:var(--text-dim);font-size:11px;">(${syms.length} symbols)</span>
        </summary>
        <table>
          <thead><tr><th>Kind</th><th>Name</th><th>Line</th><th>Signature</th></tr></thead>
          <tbody>
            ${syms.map(s => `
              <tr>
                <td><span class="badge badge-system">${esc(s.kind)}</span></td>
                <td>${esc(s.name)}</td>
                <td>${s.line_start}</td>
                <td style="font-family:var(--font-mono);font-size:11px;color:var(--text-dim);">
                  ${(s.decorators || []).length ? s.decorators.map(d => `<span style="color:var(--text-dim);">@${esc(d)}</span> `).join('') : ''}
                  ${s.parameters?.map(p => `<span style="color:var(--accent);">${esc(p.name)}</span>${p.type_annotation ? '<span style="color:var(--text-dim);">: ' + esc(p.type_annotation) + '</span>' : ''}`).join(', ') || ''}
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </details>
    `).join('')}
  `;
}

function renderDataFlow() {
  const cto = DATA.raw.cto || {};
  const flow = (cto.product_architecture || {}).data_flow || {};
  const stages = flow.pipeline_stages || [];
  return `
    <div style="overflow-x:auto;">
      <pre style="font-size:12px;">${stages.map((s, i) => `Stage ${i+1}: ${s.name}
  Input filter : ${esc(s.input_filter || '')}
  Output       : ${esc(s.output_table || s.output || '')}
  Evidence     : ${esc(s.evidence || '')}`).join('\n\n')}</pre>
    </div>
    <p style="color:var(--text-dim);font-size:12px;margin-top:8px;">
      Streaming mode: each stage polls its own PENDING_SQL independently. discover runs once; downstream stages loop until upstream finishes AND their own pending count hits 0.
      No explicit message queue — pure SQLite.
    </p>
  `;
}

// ============ RISKS ============
function renderRisks() {
  const root = document.getElementById('view-risks');
  const cto = DATA.raw.cto || {};
  const risks = cto.risk_register || [];
  const questions = cto.open_questions_carried_forward || [];
  const piConfig = DATA.raw.roles?.evidence_summary || {};

  root.innerHTML = `
    <h1>Risks & Open Questions</h1>
    <p class="subtitle">8 architectural risks identified during reverse engineering + 6 runtime validation questions.</p>

    <div class="section-title">Risk Register</div>
    <table>
      <thead><tr><th>ID</th><th>Category</th><th>Risk</th><th>Severity</th><th>Recommendation</th></tr></thead>
      <tbody>
        ${risks.map(r => `
          <tr>
            <td><code>${esc(r.risk_id)}</code></td>
            <td><span class="badge badge-system">${esc(r.category)}</span></td>
            <td><strong>${esc(r.title)}</strong><br><span style="color:var(--text-dim);font-size:11px;">${esc(r.description)}</span></td>
            <td>
              ${r.severity === 'high' ? '<span class="badge badge-needs_review">HIGH</span>' :
                r.severity === 'medium' ? '<span class="badge badge-inferred">MEDIUM</span>' :
                '<span class="badge badge-confirmed">LOW</span>'}
            </td>
            <td style="font-size:12px;">${esc(r.recommendation)}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>

    <div class="section-title">Open Questions (need runtime validation)</div>
    ${questions.map(q => `
      <details style="margin-bottom:6px;background:var(--bg-card);border:1px solid var(--border);border-radius:6px;">
        <summary style="color:var(--accent-yellow);">❓ ${esc(q.id)}: ${esc(q.question)}</summary>
        <div style="padding:10px 0;font-size:12px;color:var(--text-dim);">
          Evidence source: ${esc(q.evidence)}<br>
          Found at Stage ${q.stage}
        </div>
      </details>
    `).join('')}

    <div class="section-title">Priority Action Items</div>
    <div class="grid-cards">
      ${(cto.priority_action_items || []).map(a => `
        <div class="card" style="cursor:default;">
          <div class="card-header">
            <div class="title">P${a.priority} · ${esc(a.title)}</div>
          </div>
          <div class="meta">Risk: ${esc(a.risk_id || '')}</div>
          <div class="desc" style="color:var(--text-dim);font-size:12px;">${esc(a.impact || '')}</div>
          <div class="meta" style="margin-top:6px;">Effort: ${esc(a.effort || '')}</div>
        </div>
      `).join('')}
    </div>
  `;
}