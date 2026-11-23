import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AuditEnvironment } from './audit-env.js';
import { serveEnvironment } from './foundation-env.js';
import { manifest, strategicFeatures } from '../strategic-policy.js';

const { encodingVersion, observationSize, actionSize, historySize } = manifest;

export class StrategicEnvironment extends AuditEnvironment {
  state() {
    const state = super.state();
    if (state.terminal) {
      state.observation = Array(observationSize).fill(0);
    } else {
      const extra = strategicFeatures(this.game, state.actions);
      state.observation.push(...extra.observation);
      // Keep simulator actions unchanged; only the policy payload is extended.
      state.actions = state.actions.map((action, index) => ({
        ...action, features: [...action.features, ...extra.actionExtras[index]],
      }));
    }
    return state;
  }

  handle(request) {
    if (request.op === 'hello') return { encodingVersion, observationSize, actionSize, historySize };
    return super.handle(request);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  await serveEnvironment(new StrategicEnvironment());
}
