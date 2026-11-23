import readline from 'node:readline';
import { newGame, playCards, passTurn, returnCard, computerTurn, validSave } from '../engine.js';
import { ENCODING_VERSION, OBSERVATION_SIZE, ACTION_SIZE, observeForPolicy, legalPolicyActions, recordPolicyEvent } from '../policy.js';

let game = null;
let actions = [];
let moves = 0;
const maxActions = Number(process.env.RL_MAX_ACTIONS || 1500);
function state() {
  if (game.over) {
    actions = [];
    return { terminal: true, player: game.turn, phase: game.phase, result: game.result, observation: Array(OBSERVATION_SIZE).fill(0), actions };
  }
  actions = legalPolicyActions(game);
  return { terminal: false, player: game.turn, phase: game.phase, result: null, observation: observeForPolicy(game), actions };
}
function handle(request) {
  if (request.op === 'hello') return { encodingVersion: ENCODING_VERSION, observationSize: OBSERVATION_SIZE, actionSize: ACTION_SIZE };
  if (request.op === 'reset') {
    if (!Number.isInteger(request.seed) || request.seed < 0 || request.seed > 4294967295) throw new Error('seed must be uint32');
    if (request.level !== undefined && (!Number.isInteger(request.level) || request.level < 2 || request.level > 13)) throw new Error('level must be 2..13');
    if (request.startingSeat !== undefined && (!Number.isInteger(request.startingSeat) || request.startingSeat < 0 || request.startingSeat > 5)) throw new Error('startingSeat must be 0..5');
    let seed = request.seed;
    const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    for (let i = 0; i < 4; i++) random();
    let previous;
    if (request.continue) {
      if (!game?.over) throw new Error('Cannot continue an unfinished round');
      previous = game.result.champion ? null : game;
    } else {
      const level = request.level ?? 2 + Math.floor(random() * 12);
      const seat = request.startingSeat ?? Math.floor(random() * 6);
      previous = { round: 0, levels: [level, level], dealer: seat % 2, order: [seat] };
    }
    game = newGame(previous, random);
    moves = 0;
    if (!validSave(game)) throw new Error('Invalid reset state');
    return state();
  }
  if (request.op === 'step' || request.op === 'heuristic') {
    if (!game || game.over) throw new Error('Reset before taking actions');
    if (moves >= maxActions) throw new Error('Episode action limit exceeded');
    const player = game.turn;
    const historyLength = game.history.length;
    const phase = game.phase;
    if (request.op === 'heuristic') computerTurn(game);
    else {
      if (!Number.isInteger(request.index) || request.index < 0 || request.index >= actions.length) throw new Error('Invalid legal action index');
      const action = actions[request.index];
      if (action.kind === 'pass') passTurn(game, player);
      else if (action.kind === 'return') returnCard(game, player, action.ids[0]);
      else playCards(game, player, action.ids);
    }
    moves++;
    recordPolicyEvent(game, player, historyLength, phase);
    if (process.env.RL_CHECK_STATES === '1' && !validSave(game)) throw new Error('Invalid state after action');
    return state();
  }
  if (request.op === 'close') return { closed: true };
  throw new Error('Unknown environment operation');
}
const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  try {
    const request = JSON.parse(line);
    const response = handle(request);
    process.stdout.write(`${JSON.stringify(response)}\n`);
    if (request.op === 'close') break;
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ error: error.message })}\n`);
  }
}
