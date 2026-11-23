import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { ModelHost } from './model-host.js';
import { OBSERVATION_SIZE } from './policy.js';
import { HISTORY_SIZE } from './foundation-policy.js';

const files = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/style.css', ['style.css', 'text/css; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/engine.js', ['engine.js', 'text/javascript; charset=utf-8']],
  ['/policy.js', ['policy.js', 'text/javascript; charset=utf-8']],
  ['/foundation-policy.js', ['foundation-policy.js', 'text/javascript; charset=utf-8']],
  ['/strategic-policy.js', ['strategic-policy.js', 'text/javascript; charset=utf-8']],
  ['/training/manifests/strategic-v3.json', ['training/manifests/strategic-v3.json', 'application/json']],
  ['/joker-big.svg', ['joker-big.svg', 'image/svg+xml']],
  ['/joker-small.svg', ['joker-small.svg', 'image/svg+xml']],
]);
const model = new ModelHost();
function json(res, status, payload) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(payload));
}
async function body(req) {
  let text = '';
  let bytes = 0;
  for await (const chunk of req) {
    bytes += chunk.length;
    if (bytes > 2 * 1024 * 1024) throw new Error('Model request is too large');
    text += chunk.toString('utf8');
  }
  return JSON.parse(text);
}
const vector = (value, size) => Array.isArray(value) && value.length === size &&
  value.every(n => Number.isFinite(n) && n >= 0 && n <= 2);
const strategicObservation = value => Array.isArray(value) && value.length === model.observationSize &&
  vector(value.slice(0, OBSERVATION_SIZE), OBSERVATION_SIZE) &&
  value.slice(OBSERVATION_SIZE).every(n => Number.isFinite(n) && n >= -1 && n <= 2);
const historyVector = value => Array.isArray(value) && value.length === HISTORY_SIZE &&
  value.every(n => Number.isFinite(n) && n >= -1 && n <= 2);
const port = Number(process.env.PORT || 3000);
const server = http.createServer(async (req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname.startsWith('/api/ai/')) {
    if (!['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress) ||
      req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) {
      json(res, 403, { error: 'Model inference is restricted to local same-origin requests' });
      return;
    }
    if (pathname === '/api/ai/status' && req.method === 'GET') {
      try {
        json(res, 200, await model.start());
      } catch (error) {
        json(res, 200, { available: false, error: error.message, experimental: true });
      }
      return;
    }
    if (pathname === '/api/ai/reload' && req.method === 'POST') {
      try {
        json(res, 200, await model.reload());
      } catch (error) {
        console.error('Local model reload failed:', error.message);
        json(res, 503, { error: error.message });
      }
      return;
    }
    if (pathname === '/api/ai/score' && req.method === 'POST') {
      let request;
      try {
        if (!req.headers['content-type']?.startsWith('application/json')) throw new Error('Expected application/json');
        request = await body(req);
        if (!(model.strategic ? strategicObservation(request.observation) :
          vector(request.observation, model.observationSize)) || !Array.isArray(request.actions) ||
          request.actions.length < 1 || request.actions.length > 512 || !request.actions.every(a => vector(a, model.actionSize)) ||
          model.foundation && (!Array.isArray(request.history) || request.history.length > 4096 || !request.history.every(historyVector))) {
          throw new Error('Invalid observation or legal action features');
        }
      } catch (error) {
        if (!res.destroyed) json(res, 400, { error: error.message });
        return;
      }
      try {
        json(res, 200, { scores: await model.score(request.observation, request.actions, request.history) });
      } catch (error) {
        console.error('Local model inference failed:', error.message);
        if (!res.destroyed) json(res, 503, { error: error.message });
      }
      return;
    }
    json(res, 404, { error: 'Unknown model endpoint' });
    return;
  }
  const file = files.get(pathname);
  if (!file || !['GET', 'HEAD'].includes(req.method)) {
    res.writeHead(404).end('Not found');
    return;
  }
  try {
    const data = await readFile(new URL(file[0], import.meta.url));
    res.writeHead(200, { 'Content-Type': file[1], 'Cache-Control': 'no-cache' });
    res.end(req.method === 'HEAD' ? undefined : data);
  } catch (error) {
    console.error(error);
    res.writeHead(500).end('Unable to load game');
  }
});
for (const signal of ['SIGTERM', 'SIGINT']) process.once(signal, () => {
  model.close();
  server.close(() => process.exit(0));
});
server.listen(port, process.env.HOST || '0.0.0.0', () => console.log(`大怪路子 http://localhost:${server.address().port}`));
