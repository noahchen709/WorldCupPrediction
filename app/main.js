let teams = [];
let simulationResults = [];
let simulationReport = null;
let backtestResults = null;

function championProbability(team) {
  const strength = team.rating * 0.65 + team.attack * 0.2 + team.defense * 0.15;
  const fieldAverage =
    teams.reduce((sum, currentTeam) => sum + currentTeam.rating, 0) / teams.length;
  return Math.max(0.8, Math.pow(strength / fieldAverage, 8));
}

function normalizedTeams(region = "all") {
  if (simulationResults.length) {
    return simulationResults
      .map((result) => {
        const team = teams.find((candidate) => candidate.name === result.team);
        return {
          ...team,
          name: result.team,
          probability: result.champion_probability,
          rank: result.rank,
          elo: result.elo,
          confederation: team?.confederation || "Other",
          rating: team?.rating || result.elo
        };
      })
      .filter((team) => region === "all" || team.confederation === region)
      .sort((a, b) => b.probability - a.probability);
  }

  const visibleTeams = teams.filter((team) => region === "all" || team.confederation === region);
  const weighted = visibleTeams.map((team) => ({
    ...team,
    weight: championProbability(team)
  }));
  const total = weighted.reduce((sum, team) => sum + team.weight, 0);
  return weighted
    .map((team) => ({
      ...team,
      probability: team.weight / total
    }))
    .sort((a, b) => b.probability - a.probability);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatStage(stage) {
  const labels = {
    champion: "Champion",
    runner_up: "Runner-up",
    semifinal: "Semi-final",
    quarterfinal: "Quarter-final",
    round_of_16: "Round of 16",
    group: "Group"
  };
  return labels[stage] || stage;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderBars(region) {
  const ranked = normalizedTeams(region);
  const bars = document.querySelector("#winner-bars");

  if (!ranked.length) {
    bars.innerHTML = `<div class="team-row">No rating data loaded</div>`;
    document.querySelector("#top-team").textContent = "No data";
    document.querySelector("#top-team-detail").textContent = "Serve the project locally and refresh Elo data";
    document.querySelector("#best-strength").textContent = "No data";
    return;
  }

  const maxProbability = Math.max(...ranked.map((team) => team.probability));
  bars.innerHTML = ranked
    .map((team) => {
      const width = (team.probability / maxProbability) * 100;
      return `
        <div class="team-row">
          <div>
            <div class="team-name">${team.name}</div>
            <div class="team-meta">${team.confederation} · Elo ${team.elo || team.rating} · Rank ${team.rank || "n/a"}</div>
          </div>
          <div class="track" aria-hidden="true">
            <div class="fill" style="width: ${width}%"></div>
          </div>
          <div class="probability">${formatPercent(team.probability)}</div>
        </div>
      `;
    })
    .join("");

  const top = ranked[0];
  document.querySelector("#top-team").textContent = top.name;
  document.querySelector("#top-team-detail").textContent = `${formatPercent(top.probability)} champion probability`;

  const best = [...teams].sort((a, b) => b.rating - a.rating)[0];
  document.querySelector("#best-strength").textContent = `${best.name} ${best.elo || best.rating}`;
}

function predictMatch(home, away) {
  const ratingGap = home.rating - away.rating;
  const expected = 1 / (1 + Math.pow(10, -ratingGap / 400));
  let draw = Math.max(0.16, Math.min(0.30, 0.30 - Math.abs(ratingGap) / 1200));
  let homeWin = expected - 0.5 * draw;
  let awayWin = 1 - homeWin - draw;

  if (homeWin < 0 || awayWin < 0) {
    draw = Math.min(draw, expected * 2, (1 - expected) * 2);
    homeWin = expected - 0.5 * draw;
    awayWin = 1 - homeWin - draw;
  }

  return { homeWin, draw, awayWin };
}

function renderMatchCard() {
  if (teams.length < 2) {
    document.querySelector("#match-card").innerHTML = "";
    return;
  }

  const home = teams.find((team) => team.name === "Brazil") || teams[0];
  const away = teams.find((team) => team.name === "France") || teams[1];
  const probabilities = predictMatch(home, away);
  document.querySelector("#match-card").innerHTML = `
    <div class="versus">
      <span>${home.name}</span>
      <span>${away.name}</span>
    </div>
    <div class="prob-grid">
      <div class="prob-box">
        <span class="prob-label">${home.name}</span>
        <strong>${formatPercent(probabilities.homeWin)}</strong>
      </div>
      <div class="prob-box">
        <span class="prob-label">Draw</span>
        <strong>${formatPercent(probabilities.draw)}</strong>
      </div>
      <div class="prob-box">
        <span class="prob-label">${away.name}</span>
        <strong>${formatPercent(probabilities.awayWin)}</strong>
      </div>
    </div>
  `;
}

function renderFinishReport() {
  const summaryContainer = document.querySelector("#finish-summary");
  const tableBody = document.querySelector("#finish-table-body");

  if (!simulationReport) {
    summaryContainer.innerHTML = `
      <div class="empty-state">
        Run <code>PYTHONPATH=src python3 scripts/run_simulation.py --iterations 10000</code>
        to generate the Monte Carlo report.
      </div>
    `;
    tableBody.innerHTML = `<tr><td colspan="6">No finish probability report found</td></tr>`;
    return;
  }

  const top = simulationReport.results?.[0];
  summaryContainer.innerHTML = `
    <div class="report-stat">
      <span>Iterations</span>
      <strong>${simulationReport.iterations.toLocaleString()}</strong>
      <small>Seed ${simulationReport.seed}</small>
    </div>
    <div class="report-stat">
      <span>Field</span>
      <strong>${simulationReport.teamCount}</strong>
      <small>Teams in official 2026 format</small>
    </div>
    <div class="report-stat">
      <span>Top Champion</span>
      <strong>${escapeHtml(top?.team || "n/a")}</strong>
      <small>${top ? formatPercent(top.champion_probability) : "No result"} champion probability</small>
    </div>
    <div class="report-stat">
      <span>Generated</span>
      <strong>${new Date(simulationReport.generatedAt).toLocaleDateString()}</strong>
      <small>${escapeHtml(simulationReport.method)}</small>
    </div>
  `;

  tableBody.innerHTML = (simulationReport.results || [])
    .map((team) => {
      return `
        <tr>
          <td>
            <span class="table-team">${escapeHtml(team.team)}</span>
            <small>Elo ${team.elo.toFixed(0)} · Rank ${team.rank || "n/a"}</small>
          </td>
          <td>${formatPercent(team.champion_probability)}</td>
          <td>${formatPercent(team.final_probability)}</td>
          <td>${formatPercent(team.semifinal_probability)}</td>
          <td>${formatPercent(team.quarterfinal_probability)}</td>
          <td>${formatPercent(team.round_of_16_probability)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderGroupStageMatches() {
  const status = document.querySelector("#group-match-status");
  const container = document.querySelector("#group-matches-grid");

  if (!simulationReport || !teams.length) {
    status.textContent = "Unavailable";
    container.innerHTML = `
      <div class="empty-state">
        Run the simulation report and refresh Elo data to show group match probabilities.
      </div>
    `;
    return;
  }

  const records = new Map(teams.map((team) => [team.name, team]));
  const groups = simulationReport.tournamentStructure?.groups || [];
  const matchCount = groups.reduce(
    (count, group) => count + (group.teams.length * (group.teams.length - 1)) / 2,
    0
  );
  status.textContent = `${matchCount} matches`;
  container.innerHTML = groups
    .map((group) => {
      const rows = [];
      for (let i = 0; i < group.teams.length; i += 1) {
        for (let j = i + 1; j < group.teams.length; j += 1) {
          const homeName = group.teams[i];
          const awayName = group.teams[j];
          const home = records.get(homeName);
          const away = records.get(awayName);
          if (!home || !away) {
            continue;
          }
          const probabilities = predictMatch(home, away);
          rows.push(`
            <tr>
              <td>
                <span class="table-team">${escapeHtml(homeName)} v ${escapeHtml(awayName)}</span>
                <small>Elo ${home.rating} v ${away.rating}</small>
              </td>
              <td>${formatPercent(probabilities.homeWin)}</td>
              <td>${formatPercent(probabilities.draw)}</td>
              <td>${formatPercent(probabilities.awayWin)}</td>
            </tr>
          `);
        }
      }
      return `
        <section class="group-match-card">
          <h3>Group ${escapeHtml(group.group)}</h3>
          <div class="table-wrap">
            <table class="report-table match-table">
              <thead>
                <tr>
                  <th>Fixture</th>
                  <th>Team 1</th>
                  <th>Draw</th>
                  <th>Team 2</th>
                </tr>
              </thead>
              <tbody>${rows.join("")}</tbody>
            </table>
          </div>
        </section>
      `;
    })
    .join("");
}

function renderBacktest() {
  const status = document.querySelector("#backtest-status");
  const summaryContainer = document.querySelector("#backtest-summary");
  const tableBody = document.querySelector("#backtest-table-body");

  if (!backtestResults) {
    status.textContent = "Unavailable";
    summaryContainer.innerHTML = `
      <div class="empty-state">
        Run <code>PYTHONPATH=src python3 scripts/run_backtest.py --iterations 10000</code>
        to generate the historical comparison.
      </div>
    `;
    tableBody.innerHTML = `<tr><td colspan="7">No backtest report found</td></tr>`;
    return;
  }

  const { summary, teams: backtestTeams } = backtestResults;
  status.textContent = `${summary.iterations.toLocaleString()} simulations`;
  summaryContainer.innerHTML = `
      <div class="report-stat">
        <span>Top Pick</span>
        <strong>${escapeHtml(summary.top_pick)}</strong>
        <small>${formatPercent(summary.top_pick_probability)} · actual ${formatStage(summary.top_pick_actual_stage)}</small>
      </div>
    <div class="report-stat">
      <span>Actual Champion</span>
      <strong>${escapeHtml(summary.actual_champion)}</strong>
      <small>${formatPercent(summary.actual_champion_probability)} · model rank #${summary.actual_champion_rank}</small>
    </div>
    <div class="report-stat">
      <span>Actual Finalists</span>
      <strong>${formatPercent(summary.finalist_probability_total)}</strong>
      <small>Total probability assigned to Argentina and France reaching the final</small>
    </div>
    <div class="report-stat">
      <span>Ratings Snapshot</span>
      <strong>${summary.as_of}</strong>
      <small>Pre-tournament Elo input</small>
    </div>
  `;

  tableBody.innerHTML = backtestTeams
    .slice(0, 16)
    .map((team, index) => {
      const championClass = team.actual_stage === "champion" ? " actual-champion" : "";
      return `
        <tr class="${championClass}">
          <td>${index + 1}</td>
          <td>
            <span class="table-team">${escapeHtml(team.team)}</span>
            <small>Elo ${team.elo.toFixed(0)}</small>
          </td>
          <td>${formatStage(team.actual_stage)}</td>
          <td>${formatPercent(team.champion_probability)}</td>
          <td>${formatPercent(team.final_probability)}</td>
          <td>${formatPercent(team.semifinal_probability)}</td>
          <td>${formatPercent(team.quarterfinal_probability)}</td>
        </tr>
      `;
    })
    .join("");
}

function activateTab(tabName) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    const isActive = button.dataset.tab === tabName;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
  document.querySelectorAll("[data-tab-panel]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.tabPanel === tabName);
  });
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    activateTab(button.dataset.tab);
  });
});

document.querySelector("#region-filter").addEventListener("change", (event) => {
  renderBars(event.target.value);
});

async function loadDashboardData() {
  try {
    const response = await fetch("../data/derived_team_strengths.json");
    if (!response.ok) {
      throw new Error(`Data request failed: ${response.status}`);
    }
    const data = await response.json();
    teams = data.teams;
    document.querySelector("#data-status").textContent = `Elo as of ${data.asOf || "latest"}`;
  } catch (error) {
    document.querySelector("#data-status").textContent = "Data unavailable";
  }

  try {
    const response = await fetch("../reports/monte_carlo_results.json");
    if (response.ok) {
      const simulation = await response.json();
      simulationReport = simulation;
      simulationResults = simulation.results;
      document.querySelector("#data-status").textContent = `${simulation.iterations.toLocaleString()} simulations`;
    }
  } catch (error) {
    simulationReport = null;
    simulationResults = [];
  }

  try {
    const response = await fetch("../reports/world-cup-2022_backtest.json");
    if (response.ok) {
      backtestResults = await response.json();
    }
  } catch (error) {
    backtestResults = null;
  }

  renderFinishReport();
  renderGroupStageMatches();
  renderBars("all");
  renderBacktest();
}

loadDashboardData();
