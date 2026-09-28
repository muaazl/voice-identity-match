/**
 * VoiceGate REST API Client.
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

    async createSession() {
        const res = await fetch(`${this.baseUrl}/api/sessions/create`, { method: 'POST' });
        if (!res.ok) throw new Error(`Session creation failed: ${res.statusText}`);
        return res.json();
    }

    async enrollIdentity(name, audioBlob, sessionId) {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('file', audioBlob, `${name.replace(/\s+/g, '_')}_enroll.wav`);

        const res = await fetch(`${this.baseUrl}/api/identities/enroll`, {
            method: 'POST',
            body: formData,
            headers: { 'X-Session-ID': sessionId }
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Enrollment failed');
        }
        return res.json();
    }

    async verifySpeaker(audioBlob, sessionId, identityId = null) {
        const formData = new FormData();
        formData.append('file', audioBlob, 'verify.wav');
        if (identityId) {
            formData.append('identity_id', identityId);
        }

        const res = await fetch(`${this.baseUrl}/api/verify`, {
            method: 'POST',
            body: formData,
            headers: { 'X-Session-ID': sessionId }
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Verification failed');
        }
        return res.json();
    }
    
    async getIdentities(sessionId) {
        const res = await fetch(`${this.baseUrl}/api/identities`, {
            headers: { 'X-Session-ID': sessionId }
        });
        if (!res.ok) throw new Error(`Failed to fetch identities`);
        return res.json();
    }

    async deleteIdentity(identityId, sessionId) {
        const res = await fetch(`${this.baseUrl}/api/identities/${identityId}`, {
            method: 'DELETE',
            headers: { 'X-Session-ID': sessionId }
        });
        if (!res.ok) throw new Error(`Failed to delete identity`);
        return res.json();
    }
}

window.ApiClient = ApiClient;
