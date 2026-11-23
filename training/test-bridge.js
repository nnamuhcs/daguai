import { spawn } from 'node:child_process';
import readline from 'node:readline';

export function openEnvironment(script) {
  const child = spawn(process.execPath, [script], {
    cwd: new URL('../', import.meta.url), env: { ...process.env, RL_CHECK_STATES: '1' },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let diagnostic = '';
  child.stderr.on('data', data => { diagnostic += data; });
  child.on('error', error => { diagnostic += error.message; });
  const lines = readline.createInterface({ input: child.stdout });
  const iterator = lines[Symbol.asyncIterator]();
  return {
    async request(value) {
      let timer;
      try {
        child.stdin.write(`${JSON.stringify(value)}\n`);
        const line = await Promise.race([
          iterator.next(),
          new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Environment timeout: ${diagnostic}`)), 10000); }),
        ]);
        if (line.done) throw new Error(`Environment exited: ${diagnostic}`);
        return JSON.parse(line.value);
      } finally {
        clearTimeout(timer);
      }
    },
    close() {
      lines.close();
      child.stdin.end();
      child.kill();
    },
  };
}
