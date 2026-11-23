import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { StrategicEnvironment } from './strategic-env.js';
import { serveEnvironment } from './foundation-env.js';
import { chooseStrategicGoal } from '../strategic-goals.js';

export class GoalEnvironment extends StrategicEnvironment {
  handle(request) {
    if (request.op === 'strategic_choice') {
      if (!this.game || this.game.over) throw new Error('A strategic choice requires a live state');
      if (!Array.isArray(request.scores) || request.scores.length !== this.actions.length) {
        throw new Error('A strategic choice requires one score for each legal action');
      }
      return chooseStrategicGoal(this.game, this.actions, request.scores);
    }
    return super.handle(request);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  await serveEnvironment(new GoalEnvironment());
}
