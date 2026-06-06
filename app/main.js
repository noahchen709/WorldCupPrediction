let teams = [];
let simulationResults = [];

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
  const homeWin = 1 / (1 + Math.pow(10, -ratingGap / 400));
  const draw = Math.max(0.18, 0.28 - Math.abs(ratingGap) / 1000);
  const adjustedHomeWin = homeWin * (1 - draw);
  const awayWin = 1 - draw - adjustedHomeWin;
  return { homeWin: adjustedHomeWin, draw, awayWin };
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
      simulationResults = simulation.results;
      document.querySelector("#data-status").textContent = `${simulation.iterations.toLocaleString()} simulations`;
    }
  } catch (error) {
    simulationResults = [];
  }

  renderBars("all");
  renderMatchCard();
}

loadDashboardData();
