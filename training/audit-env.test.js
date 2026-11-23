import assert from 'node:assert/strict';
import test from 'node:test';
import { AuditEnvironment } from './audit-env.js';
import { FoundationEnvironment } from './foundation-env.js';
import { runPrivacyAudit } from './privacy-audit.js';
import { validSave } from '../engine.js';

test('audit-only operations cannot be requested through the production environment', () => {
  const production = new FoundationEnvironment();
  production.handle({ op: 'reset', seed: 20000000 });
  for (const op of ['snapshot', 'load_snapshot', 'summary', 'trace', 'advance', 'sample']) {
    assert.throws(() => production.handle({ op }), /Unknown environment operation/);
  }
});

test('audit bridge samples actual phases, resets diagnostic histories, and replays continuations', () => {
  const env = new AuditEnvironment();
  for (const phase of ['mid', 'late', 'returning', 'fresh']) {
    const state = env.handle({ op: 'sample', seed: 20000000, phase });
    assert.equal(state.terminal, false);
    assert(validSave(env.game));
    if (phase === 'returning') assert.equal(state.phase, 'returning');
    if (phase === 'fresh') assert.equal(env.trace.length, 0);
  }
  env.handle({ op: 'advance', controlled: [] });
  const summary = env.handle({ op: 'summary' });
  assert.equal(summary.actualFinishers.filter(p => p % 2 === summary.completedTeam).length, 3);
  env.handle({ op: 'reset', seed: 20000001, continue: true });
  const snapshot = env.handle({ op: 'snapshot' }).snapshot;
  env.handle({ op: 'advance', controlled: [] });
  const first = env.handle({ op: 'summary' });
  env.handle({ op: 'load_snapshot', snapshot });
  env.handle({ op: 'advance', controlled: [] });
  assert.deepEqual(env.handle({ op: 'summary' }), first);
});

test('hidden-world swaps preserve actors own/public information including asking evidence', () => {
  const result = runPrivacyAudit({ worlds: 40, positions: 4, seed: 20000000 });
  assert.equal(result.worlds, 40);
  assert.equal(result.coverage.returning, 10);
  assert.equal(result.trumpAlreadyEncoded, true);
});
