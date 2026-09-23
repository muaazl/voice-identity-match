/**
 * VoiceMimic Local Biometric Engine & Device Storage.
 * Provides client-side vector matching, EMA centroid adaptation,
 * game scoring rules, and localStorage persistence for isolated pass-and-play.
 */

(function () {
    const STORAGE_KEY = 'voicemimic_local_roster';

    // -------------------------------------------------------------
    // Vector Operations
    // -------------------------------------------------------------
    function normalizeVector(vec, eps = 1e-12) {
        let sumSq = 0;
        for (let i = 0; i < vec.length; i++) {
            sumSq += vec[i] * vec[i];
        }
        const norm = Math.sqrt(sumSq);
        if (norm > eps) {
            const out = new Float32Array(vec.length);
            for (let i = 0; i < vec.length; i++) {
                out[i] = vec[i] / norm;
            }
            return Array.from(out);
        }
        return Array.from(vec);
    }

    function cosineSimilarity(vecA, vecB) {
        if (!vecA || !vecB || vecA.length !== vecB.length) return 0.0;
        let dot = 0.0;
        for (let i = 0; i < vecA.length; i++) {
            dot += vecA[i] * vecB[i];
        }
        return Math.max(-1.0, Math.min(1.0, dot));
    }

    function updateCentroidEMA(oldCentroid, newVec, alpha = 0.85) {
        const normNew = normalizeVector(newVec);
        const combined = new Float32Array(oldCentroid.length);
        for (let i = 0; i < oldCentroid.length; i++) {
            combined[i] = alpha * oldCentroid[i] + (1.0 - alpha) * normNew[i];
        }
        return normalizeVector(Array.from(combined));
    }

    // -------------------------------------------------------------
    // Game Rules & Evaluation
    // -------------------------------------------------------------
    function identifySpeaker(queryEmbedding, players, threshold = 0.65) {
        const normQuery = normalizeVector(queryEmbedding);

        if (!players || players.length === 0) {
            return {
                winner_player_id: null,
                winner_name: null,
                confidence: 0.0,
                confidence_percent: 0.0,
                is_certain: false,
                needs_confirmation: true,
                margin: 0.0,
                rankings: [],
            };
        }

        const rawScores = players.map((p) => cosineSimilarity(normQuery, p.centroid));
        const maxScore = Math.max(...rawScores);
        const tau = 0.08;
        const expScores = rawScores.map((s) => Math.exp((s - maxScore) / tau));
        const sumExp = expScores.reduce((acc, val) => acc + val, 0);
        const probabilities = expScores.map((e) => (sumExp > 0 ? e / sumExp : 1 / players.length));

        const rankings = players.map((p, idx) => {
            const score = rawScores[idx];
            const prob = probabilities[idx];
            return {
                player_id: p.player_id,
                name: p.name,
                similarity: Math.round(score * 10000) / 10000,
                similarity_percent: Math.round(Math.max(0, score) * 1000) / 10,
                probability: Math.round(prob * 10000) / 10000,
                probability_percent: Math.round(prob * 1000) / 10,
            };
        });

        // Sort descending by similarity
        rankings.sort((a, b) => b.similarity - a.similarity);

        const top = rankings[0];
        const second = rankings.length > 1 ? rankings[1] : null;
        const margin = second ? Math.round((top.similarity - second.similarity) * 10000) / 10000 : top.similarity;
        const isCertain = (top.similarity >= threshold) && (top.probability >= 0.50 || players.length === 1);

        return {
            winner_player_id: top.player_id,
            winner_name: top.name,
            confidence: top.similarity,
            confidence_percent: top.similarity_percent,
            top_probability: top.probability,
            top_probability_percent: top.probability_percent,
            is_certain: isCertain,
            needs_confirmation: !isCertain,
            margin: margin,
            rankings: rankings,
        };
    }

    function evaluateImpostor(queryEmbedding, targetPlayer, mimicThreshold = 0.60, matchThreshold = 0.82) {
        const normQuery = normalizeVector(queryEmbedding);
        const score = cosineSimilarity(normQuery, targetPlayer.centroid);
        const roundedScore = Math.round(score * 10000) / 10000;
        const percent = Math.round(Math.max(0, score) * 1000) / 10;

        let status, message, points, securityBreached;

        if (score >= matchThreshold) {
            status = 'SYSTEM_FOOLED';
            message = 'System Fooled! Impersonation Successful';
            points = 100;
            securityBreached = true;
        } else if (score >= mimicThreshold) {
            status = 'CLOSE_MIMIC';
            message = 'Close Mimic! System Detected Difference';
            securityBreached = false;
            const span = Math.max(matchThreshold - mimicThreshold, 1e-6);
            const progress = (score - mimicThreshold) / span;
            points = Math.max(1, Math.min(99, Math.round(1 + progress * 98)));
        } else {
            status = 'POOR_ATTEMPT';
            message = 'Poor Attempt';
            points = 0;
            securityBreached = false;
        }

        return {
            target_player_id: targetPlayer.player_id,
            target_name: targetPlayer.name,
            similarity_score: roundedScore,
            similarity_percent: percent,
            status: status,
            message: message,
            points: points,
            security_breached: securityBreached,
            thresholds: {
                mimic_threshold: mimicThreshold,
                match_threshold: matchThreshold,
            },
        };
    }

    // -------------------------------------------------------------
    // Local Roster & Persistence Manager
    // -------------------------------------------------------------
    class LocalRoster {
        static getPlayers() {
            try {
                const raw = localStorage.getItem(STORAGE_KEY);
                return raw ? JSON.parse(raw) : [];
            } catch (e) {
                console.error('Failed to read from localStorage:', e);
                return [];
            }
        }

        static savePlayers(players) {
            try {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(players));
            } catch (e) {
                console.error('Failed to write to localStorage:', e);
            }
        }

        static getPlayer(playerId) {
            const players = this.getPlayers();
            return players.find((p) => p.player_id === playerId) || null;
        }

        static addPlayer(name, embedding) {
            const players = this.getPlayers();
            const playerId = `p_${Math.random().toString(36).substring(2, 10)}`;
            const normCentroid = normalizeVector(embedding);

            const newPlayer = {
                player_id: playerId,
                name: name.trim(),
                centroid: normCentroid,
                sample_count: 1,
                score: 0,
                created_at: Date.now(),
            };

            players.push(newPlayer);
            this.savePlayers(players);
            return newPlayer;
        }

        static removePlayer(playerId) {
            const players = this.getPlayers();
            const filtered = players.filter((p) => p.player_id !== playerId);
            this.savePlayers(filtered);
            return filtered;
        }

        static claimRound(playerId, queryEmbedding, points = 50, alpha = 0.85) {
            const players = this.getPlayers();
            const player = players.find((p) => p.player_id === playerId);
            if (!player) return null;

            player.centroid = updateCentroidEMA(player.centroid, queryEmbedding, alpha);
            player.sample_count = (player.sample_count || 1) + 1;
            player.score = (player.score || 0) + points;

            this.savePlayers(players);
            return player;
        }

        static addScore(playerId, points) {
            const players = this.getPlayers();
            const player = players.find((p) => p.player_id === playerId);
            if (!player) return 0;

            player.score = (player.score || 0) + points;
            this.savePlayers(players);
            return player.score;
        }

        static resetScoresOnly() {
            const players = this.getPlayers();
            players.forEach((p) => {
                p.score = 0;
            });
            this.savePlayers(players);
            return players;
        }

        static clearAll() {
            localStorage.removeItem(STORAGE_KEY);
            return [];
        }

        static getScoreboard() {
            const players = this.getPlayers();
            return [...players].sort((a, b) => b.score - a.score);
        }
    }

    // Expose to global window object
    window.LocalEngine = {
        normalizeVector,
        cosineSimilarity,
        updateCentroidEMA,
        identifySpeaker,
        evaluateImpostor,
        LocalRoster,
    };
})();
