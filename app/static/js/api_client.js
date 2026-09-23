/**
 * VoiceMimic REST API Client.
 * Connects the frontend to FastAPI backend endpoints.
 */

class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async getHealth() {
        const res = await fetch(`${this.baseUrl}/api/health`);
        if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
        return res.json();
    }

    async registerPlayer(playerName, audioBlob) {
        const formData = new FormData();
        formData.append('player_name', playerName);
        formData.append('file', audioBlob, `${playerName.replace(/\s+/g, '_')}_reg.wav`);

        const res = await fetch(`${this.baseUrl}/api/players/register`, {
            method: 'POST',
            body: formData,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Registration failed');
        }
        return res.json();
    }

    async listPlayers() {
        const res = await fetch(`${this.baseUrl}/api/players`);
        if (!res.ok) throw new Error(`Failed to list players: ${res.statusText}`);
        return res.json();
    }

    async deletePlayer(playerId) {
        const res = await fetch(`${this.baseUrl}/api/players/${playerId}`, {
            method: 'DELETE',
        });
        if (!res.ok) throw new Error(`Failed to delete player: ${res.statusText}`);
        return res.json();
    }

    async guessWho(audioBlob, threshold = 0.65) {
        const formData = new FormData();
        formData.append('threshold', threshold.toString());
        formData.append('file', audioBlob, 'query_speaker.wav');

        const res = await fetch(`${this.baseUrl}/api/game/guess-who`, {
            method: 'POST',
            body: formData,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Identification failed');
        }
        return res.json();
    }

    async confirmSpeaker(actualPlayerId, roundId = null, points = 50) {
        const res = await fetch(`${this.baseUrl}/api/game/confirm-speaker`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                actual_player_id: actualPlayerId,
                round_id: roundId,
                points: points,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Speaker confirmation failed');
        }
        return res.json();
    }

    async mimicChallenge(targetPlayerId, audioBlob, mimicThreshold = 0.60, matchThreshold = 0.82) {
        const formData = new FormData();
        formData.append('target_player_id', targetPlayerId);
        formData.append('mimic_threshold', mimicThreshold.toString());
        formData.append('match_threshold', matchThreshold.toString());
        formData.append('file', audioBlob, 'mimic_attempt.wav');

        const res = await fetch(`${this.baseUrl}/api/game/mimic-challenge`, {
            method: 'POST',
            body: formData,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Mimic evaluation failed');
        }
        return res.json();
    }

    async getScoreboard() {
        const res = await fetch(`${this.baseUrl}/api/game/scoreboard`);
        if (!res.ok) throw new Error(`Failed to load scoreboard: ${res.statusText}`);
        return res.json();
    }

    async resetGame(resetScoresOnly = false) {
        const res = await fetch(`${this.baseUrl}/api/game/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reset_scores_only: resetScoresOnly }),
        });
        if (!res.ok) throw new Error(`Reset failed: ${res.statusText}`);
        return res.json();
    }
}

window.ApiClient = ApiClient;
