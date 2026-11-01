#!/usr/bin/env node
/**
 * ============================================================================
 * KBS INDUSTRIAL FAULT DIAGNOSIS - ONE-COMMAND LOCAL LAUNCHER
 * ----------------------------------------------------------------------------
 * Run with:  npm run dev
 *
 * This script performs the full local startup sequence for the existing
 * Flask application (app.py) without modifying any application source code:
 *
 *   1. Locates a usable Python 3 interpreter (py launcher or python on PATH).
 *   2. Creates submission/venv if it does not already exist.
 *   3. Installs missing backend dependencies from submission/requirements.txt.
 *   4. Detects and clears a stale instance holding the port.
 *   5. Starts the Flask backend (which also serves the frontend/templates).
 *   6. Waits until the server genuinely answers, then prints the URL banner.
 *   7. Forwards all server logs and shuts everything down on CTRL+C.
 *
 * The project is a single Flask service: app.py serves the REST API and the
 * UI (templates/index.html + static/app.js) from the same origin, so there is
 * no separate frontend dev server to orchestrate.
 * ============================================================================
 */

'use strict';

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

// ---------------------------------------------------------------------------
// Paths and configuration
// ---------------------------------------------------------------------------

// This file lives in <project root>/scripts, so the project root is its parent.
const ROOT_DIR = path.resolve(__dirname, '..');
const SUBMISSION_DIR = path.join(ROOT_DIR, 'submission');
const VENV_DIR = path.join(SUBMISSION_DIR, 'venv');
const REQ_FILE = path.join(SUBMISSION_DIR, 'requirements.txt');
const APP_FILE = path.join(SUBMISSION_DIR, 'app.py');

const HOST = '0.0.0.0';
const BROWSE_HOST = 'localhost';
const PORT = 5000;

// The Flask app must be able to import these to start.
const REQUIRED_MODULES = ['flask'];

// ---------------------------------------------------------------------------
// Small console helpers
// ---------------------------------------------------------------------------

const isTTY = process.stdout.isTTY;
const paint = (code, text) => (isTTY ? `\u001b[${code}m${text}\u001b[0m` : text);
const bold = (t) => paint('1', t);
const dim = (t) => paint('2', t);
const green = (t) => paint('32', t);
const red = (t) => paint('31', t);
const yellow = (t) => paint('33', t);
const cyan = (t) => paint('36', t);

const step = (msg) => console.log(`${cyan('[setup]')} ${msg}`);
const ok = (msg) => console.log(`${green('[ ok  ]')} ${msg}`);
const warn = (msg) => console.log(`${yellow('[warn ]')} ${msg}`);

function fail(msg, hintLines = []) {
  console.error('');
  console.error(red(bold('  SETUP FAILED  ')) + ' ' + msg);
  for (const line of hintLines) console.error('  ' + line);
  console.error('');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Python interpreter discovery
// ---------------------------------------------------------------------------

const PYTHON_CANDIDATES = [
  { cmd: 'py', args: ['-3.14'], label: 'py -3.14' },
  { cmd: 'py', args: ['-3.13'], label: 'py -3.13' },
  { cmd: 'py', args: ['-3.12'], label: 'py -3.12' },
  { cmd: 'py', args: ['-3.11'], label: 'py -3.11' },
  { cmd: 'python', args: [], label: 'python' },
  { cmd: 'python3', args: [], label: 'python3' },
];

function findPython() {
  for (const candidate of PYTHON_CANDIDATES) {
    const probe = spawnSync(
      candidate.cmd,
      [...candidate.args, '-c', 'import sys; print(sys.version_info[0], sys.version_info[1])'],
      { encoding: 'utf8', windowsHide: true }
    );

    if (probe.error || probe.status !== 0) continue;

    const [major, minor] = String(probe.stdout).trim().split(/\s+/).map(Number);
    if (major === 3 && minor >= 8) {
      return { ...candidate, version: `${major}.${minor}` };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Port occupancy handling
// ---------------------------------------------------------------------------

function findListenersOnPort(port) {
  const result = spawnSync('netstat', ['-ano'], { encoding: 'utf8', windowsHide: true });
  if (result.error || result.status !== 0 || !result.stdout) return [];

  const pids = new Set();
  for (const rawLine of result.stdout.split(/\r?\n/)) {
    const parts = rawLine.trim().split(/\s+/);
    if (parts.length < 5) continue;
    const [proto, local, , state, pid] = parts;
    if (!/^TCP/i.test(proto)) continue;
    if (state.toUpperCase() !== 'LISTENING') continue;
    if (!local.endsWith(':' + port)) continue;
    if (/^\d+$/.test(pid)) pids.add(pid);
  }
  return [...pids];
}

function processNameFor(pid) {
  const result = spawnSync('tasklist', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'], {
    encoding: 'utf8',
    windowsHide: true,
  });
  if (result.error || !result.stdout) return '';
  const match = result.stdout.match(/^"([^"]+)"/m);
  return match ? match[1] : '';
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Ask whatever is listening on `port` whether it is this KBS application.
 * Resolves to the parsed machine list, or null when nothing recognisable answers.
 */
function probeKbsApp(port) {
  return new Promise((resolve) => {
    const req = http.get(
      { host: '127.0.0.1', port, path: '/api/machines', timeout: 1500 },
      (res) => {
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => { body += chunk; });
        res.on('end', () => {
          if (res.statusCode !== 200) return resolve(null);
          try {
            const data = JSON.parse(body);
            const looksLikeKbs =
              Array.isArray(data) &&
              data.length > 0 &&
              data.every((m) => m && typeof m.machine_type === 'string');
            resolve(looksLikeKbs ? data : null);
          } catch {
            resolve(null);
          }
        });
      }
    );
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}

/**
 * Wait until nothing live is listening on the port.
 * A listener whose PID no longer resolves is a stale TCP-table entry, not a
 * running service, so it does not count as occupied.
 */
async function waitForPortRelease(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const live = findListenersOnPort(port).filter((pid) => processNameFor(pid) !== '');
    if (live.length === 0 && (await probeKbsApp(port)) === null) return true;
    if (Date.now() > deadline) return false;
    await sleep(250);
  }
}

/**
 * Prepare the port for a new server.
 *
 * Returns 'reuse' when this KBS app already answers on the port (nothing to
 * start), or 'free' when the port is ready for a new server.
 */
async function ensurePortAvailable(port) {
  // A previous instance may still be running - most often because the Flask
  // debug reloader outlived the terminal it was started from. Reuse it rather
  // than killing something the user may be actively using.
  if ((await probeKbsApp(port)) !== null) {
    step(`Port ${port} already serves this KBS application - reusing the running instance.`);
    return 'reuse';
  }

  const pids = findListenersOnPort(port);
  if (pids.length === 0) return 'free';

  const foreign = [];

  for (const pid of pids) {
    const name = processNameFor(pid);

    // Unresolvable PID: stale entry in the TCP table left by an exited process.
    // Ask Windows to clean it up; it does not own the port any more.
    if (!name) {
      spawnSync('taskkill', ['/PID', pid, '/T', '/F'], { encoding: 'utf8', windowsHide: true });
      continue;
    }

    // Only reclaim the port from a previous Python run of this app. Anything
    // else on the port is left alone and reported instead of being killed.
    if (!/^python/i.test(name)) {
      foreign.push({ pid, name });
      continue;
    }

    step(`Port ${port} held by a previous Python server (${name}, PID ${pid}) - stopping it.`);
    const killed = spawnSync('taskkill', ['/PID', pid, '/T', '/F'], {
      encoding: 'utf8',
      windowsHide: true,
    });
    if (killed.status !== 0) {
      warn(`Could not stop PID ${pid}; re-checking the port.`);
    }
  }

  if (foreign.length > 0) {
    const list = foreign.map((entry) => `"${entry.name}" (PID ${entry.pid})`).join(', ');
    fail(
      `Port ${port} is already in use by ${list}, which is not a Python server.`,
      [
        'Stop that process yourself, or change the port in app.py and scripts/dev.js.',
        `To inspect it:  tasklist /FI "PID eq ${foreign[0].pid}"`,
      ]
    );
  }

  if (await waitForPortRelease(port, 8000)) return 'free';

  // Still busy after the kill attempts: if it is our own app, simply reuse it.
  if ((await probeKbsApp(port)) !== null) {
    step(`Port ${port} already serves this KBS application - reusing the running instance.`);
    return 'reuse';
  }

  warn(`Port ${port} still reports a listener; starting anyway.`);
  return 'free';
}

// ---------------------------------------------------------------------------
// Environment bootstrap: venv + dependencies
// ---------------------------------------------------------------------------

function venvPython() {
  const candidate = path.join(VENV_DIR, 'Scripts', 'python.exe');
  return fs.existsSync(candidate) ? candidate : null;
}

function ensureVenv(basePython) {
  const existing = venvPython();
  if (existing) {
    ok(`Virtual environment ready ${dim('(' + path.relative(ROOT_DIR, existing) + ')')}`);
    return existing;
  }

  step(`Creating virtual environment using ${basePython.label} ...`);
  const created = spawnSync(basePython.cmd, [...basePython.args, '-m', 'venv', VENV_DIR], {
    cwd: SUBMISSION_DIR,
    stdio: 'inherit',
    windowsHide: true,
  });

  if (created.error || created.status !== 0) {
    fail(
      'Could not create the Python virtual environment.',
      [
        'Try running manually:',
        `  ${basePython.label} -m venv "${VENV_DIR}"`,
        'If the venv module is unavailable, install the full Python distribution:',
        '  https://www.python.org/downloads/windows/',
      ]
    );
  }

  const interpreter = venvPython();
  if (!interpreter) fail('Virtual environment was created but Scripts\\python.exe is missing.');
  ok('Virtual environment created.');
  return interpreter;
}

function hasAllModules(pythonExe, modules) {
  const check = modules.map((m) => `import ${m}`).join('; ');
  const probe = spawnSync(pythonExe, ['-c', check], { encoding: 'utf8', windowsHide: true });
  return probe.status === 0;
}

function ensureDependencies(pythonExe) {
  if (hasAllModules(pythonExe, REQUIRED_MODULES)) {
    ok(`Backend dependencies present ${dim('(flask)')}`);
    return;
  }

  if (!fs.existsSync(REQ_FILE)) {
    fail(`requirements.txt not found at ${REQ_FILE}`);
  }

  step('Installing backend dependencies (this may take a moment on first run) ...');
  const installed = spawnSync(
    pythonExe,
    ['-m', 'pip', 'install', '--disable-pip-version-check', '-r', REQ_FILE],
    { cwd: SUBMISSION_DIR, stdio: 'inherit', windowsHide: true }
  );

  if (installed.error || installed.status !== 0) {
    fail(
      'Dependency installation failed.',
      [
        'Check your internet connection, then retry:',
        `  "${pythonExe}" -m pip install -r "${REQ_FILE}"`,
      ]
    );
  }

  if (!hasAllModules(pythonExe, REQUIRED_MODULES)) {
    fail('Dependencies installed but Flask still cannot be imported.');
  }
  ok('Backend dependencies installed.');
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

function waitForServer(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;

  return new Promise((resolve) => {
    const attempt = () => {
      if (Date.now() > deadline) return resolve(false);

      const req = http.get(
        { host: '127.0.0.1', port, path: '/api/machines', timeout: 2000 },
        (res) => {
          res.resume();
          if (res.statusCode === 200) return resolve(true);
          setTimeout(attempt, 400);
        }
      );

      req.on('error', () => setTimeout(attempt, 400));
      req.on('timeout', () => {
        req.destroy();
        setTimeout(attempt, 400);
      });
    };

    attempt();
  });
}

function printBanner() {
  const base = `http://${BROWSE_HOST}:${PORT}`;
  const lines = [
    '',
    green('========================================'),
    green(bold('       APPLICATION STARTED SUCCESSFULLY')),
    green('========================================'),
    '',
    `  ${bold('Dashboard (UI + API):')}  ${cyan(base + '/')}`,
    `  ${bold('Backend server:')}       ${cyan(base)}`,
    `  ${bold('API base:')}             ${cyan(base + '/api')}`,
    '',
    `  Endpoints:`,
    dim('    GET  /api/machines                          loaded machinery & stats'),
    dim('    GET  /api/machines/<machine>/symptoms       observable symptoms'),
    dim('    GET  /api/machines/<machine>/components     subsystem components'),
    dim('    GET  /api/sensor-rules                      AI4I 2020 telemetry rules'),
    dim('    POST /api/diagnose                          forward-chaining diagnosis'),
    '',
    `  ${dim('Machines: motor | pump | gearbox | compressor')}`,
    '',
    '  Both the frontend and backend are served by this one process.',
    '',
    yellow('  Press CTRL+C to stop all services.'),
    green('========================================'),
    '',
  ];
  console.log(lines.join('\n'));
}

function openBrowser(url) {
  // Best-effort only: a browser that will not open must never break startup.
  const command =
    process.platform === 'win32' ? { cmd: 'cmd', args: ['/c', 'start', '', url] }
    : process.platform === 'darwin' ? { cmd: 'open', args: [url] }
    : { cmd: 'xdg-open', args: [url] };

  try {
    const child = spawn(command.cmd, command.args, {
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
    });
    child.on('error', () => {});
    child.unref();
  } catch {
    // Ignored: the URL is printed above regardless.
  }
}

async function main() {
  console.log('');
  console.log(bold('  KBS Industrial Fault Diagnosis') + dim('  -  starting local environment'));
  console.log('');

  // --- Sanity: project layout -------------------------------------------
  if (!fs.existsSync(APP_FILE)) {
    fail(`Could not find the application at ${APP_FILE}`);
  }

  // --- Python -------------------------------------------------------------
  step('Locating a Python 3 interpreter ...');
  const basePython = findPython();
  if (!basePython) {
    fail(
      'No Python 3.8+ interpreter was found on this machine.',
      [
        'Install Python 3 from:',
        '  https://www.python.org/downloads/windows/',
        '',
        'During installation, tick "Add python.exe to PATH".',
        'Then re-open the VS Code terminal and run:  npm run dev',
      ]
    );
  }
  ok(`Found Python ${basePython.version} ${dim('(' + basePython.label + ')')}`);

  // --- Virtual environment ------------------------------------------------
  const pythonExe = ensureVenv(basePython);

  // --- Dependencies -------------------------------------------------------
  ensureDependencies(pythonExe);

  // --- Port ---------------------------------------------------------------
  const portState = await ensurePortAvailable(PORT);

  if (portState === 'reuse') {
    printBanner();
    if (!process.env.NO_OPEN) openBrowser(`http://${BROWSE_HOST}:${PORT}/`);
    console.log(yellow('  An existing server is handling this URL. Press CTRL+C to exit this launcher.'));
    return;
  }

  // --- Start the server ---------------------------------------------------
  step(`Starting Flask server on ${HOST}:${PORT} ...`);

  const server = spawn(pythonExe, [APP_FILE], {
    cwd: SUBMISSION_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    windowsHide: true,
  });

  let shuttingDown = false;

  const forward = (stream, target) => {
    let buffer = '';
    stream.on('data', (chunk) => {
      buffer += chunk.toString();
      const parts = buffer.split(/\r?\n/);
      buffer = parts.pop();
      for (const line of parts) target.write(dim('  [server] ') + line + '\n');
    });
  };

  forward(server.stdout, process.stdout);
  forward(server.stderr, process.stderr);

  const shutdown = (code = 0) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log('');
    step('Stopping the Flask server ...');
    if (server.pid) {
      // /T kills the whole tree, which includes the Flask debug reloader child.
      spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], {
        windowsHide: true,
        stdio: 'ignore',
      });
    }
    ok('All services stopped.');
    process.exit(code);
  };

  process.on('SIGINT', () => shutdown(0));
  process.on('SIGTERM', () => shutdown(0));

  let serverReady = false;

  server.on('exit', (code, signal) => {
    if (shuttingDown) return;
    console.log('');
    const when = serverReady ? 'after it had started' : 'before it was ready';
    if (signal) {
      fail(`The Flask server was terminated ${when} (signal ${signal}).`);
    }
    fail(
      `The Flask server exited with code ${code} ${when}.`,
      ['Scroll up to the [server] lines above for the underlying Python error.']
    );
  });

  // --- Readiness ----------------------------------------------------------
  step('Waiting for the server to answer ...');
  const ready = await waitForServer(PORT, 60000);

  if (!ready) {
    fail(
      `The server did not respond on port ${PORT} within 60 seconds.`,
      ['Scroll up to the [server] lines above for the underlying Python error.']
    );
  }

  printBanner();
  serverReady = true;

  // Hand the user straight to the running app. Opt out with NO_OPEN=1.
  if (!process.env.NO_OPEN) openBrowser(`http://${BROWSE_HOST}:${PORT}/`);

  // Keep the process alive for as long as the server runs.
  await new Promise((resolve) => server.on('exit', resolve));
}

main().catch((error) => {
  fail(error && error.stack ? error.stack : String(error));
});
