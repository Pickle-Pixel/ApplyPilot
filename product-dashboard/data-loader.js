const PI_ROOT = '../.project-intelligence';

async function loadAll() {
  const endpoints = {
    roles:     `${PI_ROOT}/roles/user_roles.json`,
    journeys:  `${PI_ROOT}/journeys/user_journeys.json`,
    features:  `${PI_ROOT}/features/business_features.json`,
    caps:      `${PI_ROOT}/capabilities/atomic_capabilities.json`,
    symbols:   `${PI_ROOT}/inventory/symbols.json`,
    files:     `${PI_ROOT}/inventory/files.json`,
    cto:       `${PI_ROOT}/dashboard/cto_dashboard.json`,
  };
  const out = {};
  for (const [k, v] of Object.entries(endpoints)) {
    try {
      const r = await fetch(v);
      out[k] = await r.json();
    } catch (e) {
      console.error(`Failed to load ${k} from ${v}`, e);
      out[k] = null;
    }
  }
  return buildIndexes(out);
}

function buildIndexes(raw) {
  const F = {};
  if (raw.files && raw.files.files) {
    for (const f of raw.files.files) F[f.file_id] = f.path || f.file_path || '';
  }

  const S = {};
  if (raw.symbols && raw.symbols.symbols) {
    for (const s of raw.symbols.symbols) S[s.symbol_id] = s;
  }

  const C = {};
  if (raw.caps && raw.caps.capabilities) {
    for (const c of raw.caps.capabilities) C[c.capability_id] = c;
  }

  const FE = {};
  if (raw.features && raw.features.features) {
    for (const f of raw.features.features) FE[f.feature_id] = f;
  }

  const J = {};
  if (raw.journeys && raw.journeys.journeys) {
    for (const j of raw.journeys.journeys) J[j.journey_id] = j;
  }

  return { raw, F, S, C, FE, J };
}

function resolveFileFromSymbol(sym, F) {
  if (!sym) return null;
  const fid = sym.file_id;
  return F[fid] || fid;
}

function findSymbolsForCapability(cap, S, F) {
  const out = [];
  const symIds = cap.source_symbols || [];
  for (const sid of symIds) {
    const sym = S[sid];
    if (!sym) continue;
    out.push({
      symbol_id: sid,
      kind: sym.kind,
      name: sym.name,
      qualified_name: sym.qualified_name,
      line: sym.line_start,
      file: resolveFileFromSymbol(sym, F),
      decorators: sym.decorators || [],
      parameters: sym.parameters || [],
    });
  }
  return out;
}

function findSymbolsForFeature(feature, S, F) {
  const out = [];
  const seen = new Set();
  const entryCandidates = feature.entry_candidates || [];
  for (const e of entryCandidates) {
    const parts = e.split(':');
    const filePart = parts[0] || '';
    const symName = parts[1] || '';
    for (const s of Object.values(S)) {
      if (seen.has(s.symbol_id)) continue;
      const sFile = resolveFileFromSymbol(s, F) || '';
      const nameMatch = symName ? s.name === symName.split('.').pop() : true;
      const fileMatch = filePart ? sFile.replace(/\\/g,'/').endsWith(filePart.replace(/\\/g,'/')) : true;
      if (nameMatch && fileMatch) {
        out.push({
          symbol_id: s.symbol_id,
          kind: s.kind,
          name: s.name,
          qualified_name: s.qualified_name,
          line: s.line_start,
          file: resolveFileFromSymbol(s, F),
        });
        seen.add(s.symbol_id);
      }
    }
  }
  return out;
}

function parseEvidenceString(raw) {
  if (!raw) return null;
  const re = /^(.+?)[:：](\d+(?:-\d+)?)\s*[—\-]\s*(.+)$/;
  const m = raw.match(re);
  if (m) {
    return { file: m[1].trim(), lines: m[2], desc: m[3].trim() };
  }
  const re2 = /^(.+?)[:：](\d+(?:-\d+)?)$/;
  const m2 = raw.match(re2);
  if (m2) {
    return { file: m2[1].trim(), lines: m2[2], desc: '' };
  }
  return { file: '', lines: '', desc: raw };
}

function parseTriggerString(raw) {
  if (!raw) return { primary: '', chain: [] };
  const parts = raw.split('→').map(s => s.trim());
  return { primary: parts[0] || '', chain: parts.slice(1) };
}

function capabilitiesForStep(step, FE, C) {
  const fids = step.feature_ids || [];
  const out = [];
  const seen = new Set();
  for (const fid of fids) {
    const feat = FE[fid];
    if (!feat) continue;
    const cids = feat.capability_ids || [];
    for (const cid of cids) {
      if (seen.has(cid)) continue;
      const cap = C[cid];
      if (cap) { out.push(cap); seen.add(cid); }
    }
  }
  return out;
}

function featuresForStep(step, FE) {
  return (step.feature_ids || []).map(id => FE[id]).filter(Boolean);
}

function journeysByTier(journeys, tier) {
  if (!tier || tier === 0) return journeys;
  return journeys.filter(j => (j.tier_required || 1) === tier);
}

window.PI = {
  loadAll, buildIndexes,
  resolveFileFromSymbol, findSymbolsForCapability, findSymbolsForFeature,
  parseEvidenceString, parseTriggerString,
  capabilitiesForStep, featuresForStep, journeysByTier,
};