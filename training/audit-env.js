import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FoundationEnvironment, serveEnvironment } from './foundation-env.js';
import { validSave, publicCounts, publicPolicyHistory, classify } from '../engine.js';

export class AuditEnvironment extends FoundationEnvironment {
  constructor(...args) {
    super(...args);
    this.trace = [];
    this.diagnostics = [];
  }

  takeStep(request) {
    const game = this.game;
    const actor = game.turn;
    const before = game.hands[actor];
    const phase = game.phase;
    const historyLength = game.history.length;
    const prefix = publicPolicyHistory(game).length;
    const holder = game.lastPlayer;
    const leading = game.target === null;
    const action = request.op === 'step' ? this.actions[request.index] : null;
    super.takeStep(request);
    const play = game.history.length > historyLength ? game.history.at(-1) : null;
    const after = game.hands[actor];
    this.trace.push({ op: request.op, index: request.index, actor, phase, prefix,
      action: action ? { kind: action.kind, ids: action.ids, target: action.target } : null });
    this.diagnostics.push({
      actor, kind: phase === 'returning' ? 'return' : action?.kind === 'ask' ? 'ask' : play ? 'play' : 'pass',
      cards: play?.cards.length ?? 0, jokerCount: play?.cards.filter(c => c.rank > 14).length ?? 0,
      category: play?.type.category ?? null,
      jokerPlayValue: play?.cards.some(c => c.rank > 14) ? play.type.values[0] : null,
      allyOverride: Boolean(play && !leading && holder % 2 === actor % 2),
      receiveWithLegalRemainder: Boolean(play && !leading && holder % 2 === actor % 2 &&
        after.length > 0 && classify(after, game.level)),
      beforeCount: before.length, afterCount: after.length,
      trickWinner: !leading && game.target === null && !game.over ? holder : null,
    });
  }

  summary() {
    const game = this.game;
    if (!game?.over) throw new Error('Public terminal summary requires a finished round');
    const actual = game.order.slice(0, game.result.finishedCount);
    const completedTeam = actual.at(-1) % 2;
    if (actual.filter(seat => seat % 2 === completedTeam).length !== 3) throw new Error('Invalid genuine finisher count');
    return { ...game.result, order: game.order, actualFinishers: actual,
      headTeam: game.order[0] % 2, completedTeam, remainingCounts: publicCounts(game),
      diagnostics: this.diagnostics, moves: this.moves, level: game.level, dealer: game.dealer,
      tributeCount: game.exchanges.length, publicHistoryLength: publicPolicyHistory(game).length };
  }

  advance(controlled) {
    if (!Array.isArray(controlled) || controlled.some(p => !Number.isInteger(p) || p < 0 || p > 5)) throw new Error('Invalid controlled seats');
    while (!this.game.over && !controlled.includes(this.game.turn)) this.takeStep({ op: 'heuristic' });
    return this.state();
  }

  handle(request) {
    if (request.op === 'snapshot') {
      if (!this.game) throw new Error('No game to snapshot');
      return { snapshot: structuredClone(this.game) };
    }
    if (request.op === 'load_snapshot') {
      if (!validSave(request.snapshot)) throw new Error('Invalid continuation snapshot');
      this.game = structuredClone(request.snapshot);
      this.moves = 0;
      this.historyOffset = 0;
      this.trace = [];
      this.diagnostics = [];
      return this.state();
    }
    if (request.op === 'summary') return this.summary();
    if (request.op === 'trace') return { trace: this.trace };
    if (request.op === 'advance') return this.advance(request.controlled);
    if (request.op === 'agent_step') {
      this.takeStep({ op: 'step', index: request.index });
      return this.advance(request.controlled);
    }
    if (request.op === 'sample') {
      if (!['fresh', 'mid', 'late', 'returning'].includes(request.phase)) throw new Error('Invalid sample phase');
      super.handle({ op: 'reset', seed: request.seed });
      this.trace = [];
      this.diagnostics = [];
      if (request.phase === 'mid') {
        for (let i = 0; i < 1 + request.seed % 48 && !this.game.over; i++) this.takeStep({ op: 'heuristic' });
      } else if (request.phase === 'late') {
        while (!this.game.over && this.game.hands[this.game.turn].length > 10) this.takeStep({ op: 'heuristic' });
      } else if (request.phase === 'returning') {
        let round = 0;
        while (this.game.phase !== 'returning') {
          if (++round > 20) throw new Error('Could not obtain an authentic tribute continuation');
          while (!this.game.over) this.takeStep({ op: 'heuristic' });
          super.handle({ op: 'reset', seed: (request.seed + round) >>> 0, continue: true });
          this.trace = [];
          this.diagnostics = [];
        }
      }
      if (this.game.over) throw new Error('Sample ended before an actor decision');
      if (!validSave(this.game)) throw new Error('Invalid sampled position');
      this.historyOffset = 0;
      return this.state();
    }
    if (request.op === 'reset') {
      this.trace = [];
      this.diagnostics = [];
    }
    return super.handle(request);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  await serveEnvironment(new AuditEnvironment());
}
