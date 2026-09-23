/**
 * VoiceMimic REST API Client.
 * Connects the frontend to stateless FastAPI backend endpoints.
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

    /**
     * Stateless Audio Feature Extraction.
     * Takes raw audio bytes, executes DTLN + Silero VAD + CAM++ on server,
     * and returns the 192-D embedding vector and acoustic metrics.
     */
    async extractEmbedding(audioBlob) {
        const formData = new FormData();
        formData.append('file', audioBlob, 'audio_sample.wav');

        const res = await fetch(`${this.baseUrl}/api/audio/embed`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Audio embedding extraction failed');
        }
        return res.json();
    }

    /**
     * Backward-compatible registration endpoint.
     */
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
}

window.ApiClient = ApiClient;
