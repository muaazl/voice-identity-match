/**
 * VoiceMimic SPA Controller & State Machine.
 * Device-local pass-and-play party game controller.
 * Roster and scores persist in localStorage; audio embeddings are processed statelessly by the backend.
 */

document.addEventListener('DOMContentLoaded', () => {
    const api = new ApiClient();
    const recorder = new AudioRecorder();
    const sounds = new SoundEffects();

    // -------------------------------------------------------------
    // App State
    // -------------------------------------------------------------
    const state = {
        players: [],
        activeMode: 'guess_who', // 'guess_who' | 'impostor'
        isRecording: false,
        recordingType: null, // 'reg' | 'guess' | 'mimic'
        currentRound: null,
    };

    // -------------------------------------------------------------
    // Suggested Speech Prompts (Calibrated for ~10 seconds of speech: 22-25 words)
    // -------------------------------------------------------------
    const SUGGESTED_PHRASES = [
        "The quick brown fox jumped over the lazy sleeping dogs while the bright morning sun began to warm up the quiet forest trail.",
        "My unique voiceprint serves as my biometric passport today, allowing the neural network to analyze my natural tone, pitch, and acoustic cadence accurately.",
        "Artificial intelligence listens closely to every harmonic frequency in human speech to distinguish between genuine friends and clever impostors in this biometric game.",
        "Sphinx of black quartz, please judge my vocal vow as I speak clearly into the microphone so the machine learns my voice.",
        "To travel across galaxies and explore uncharted stars, we must first master the art of clear communication and learn to trust each other.",
        "Every great mystery begins with a quiet whisper in the shadows, waiting for someone perceptive enough to uncover the hidden truth beneath.",
        "When the cool autumn wind sweeps through the golden valley, old memories return of warm campfires, laughter, and stories shared among lifelong friends.",
        "Technology evolves at incredible speed, yet nothing is more captivating than how unique frequencies of sound can identify who is speaking without seeing them."
    ];
    let phraseIndex = 0;

    // -------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------
    // Header
    const enrolledCountEl = document.getElementById('enrolled-count');
    const btnToggleSound = document.getElementById('btn-toggle-sound');
    const btnOpenGuide = document.getElementById('btn-open-guide');
    const btnOpenLeaderboard = document.getElementById('btn-open-leaderboard');
    const btnOpenReset = document.getElementById('btn-open-reset');

    // Segmented Mode Switch
    const tabGuessWho = document.getElementById('tab-guess-who');
    const tabImpostor = document.getElementById('tab-impostor');
    const sectionGuessWho = document.getElementById('section-guess-who');
    const sectionImpostor = document.getElementById('section-impostor');

    // Registration & Suggested Phrase
    const inputPlayerName = document.getElementById('input-player-name');
    const btnRecordReg = document.getElementById('btn-record-reg');
    const textRecordReg = document.getElementById('text-record-reg');
    const textSuggestedPhrase = document.getElementById('text-suggested-phrase');
    const btnShufflePhrase = document.getElementById('btn-shuffle-phrase');
    const rosterEmptyEl = document.getElementById('roster-empty');
    const rosterChipsEl = document.getElementById('roster-chips');

    // Arena: Guess Who
    const btnRecordGuess = document.getElementById('btn-record-guess');
    const textRecordGuess = document.getElementById('text-record-guess');
    const canvasGuess = document.getElementById('canvas-guess-waveform');
    const cardGuessResult = document.getElementById('card-guess-result');
    const textPredictedName = document.getElementById('text-predicted-name');
    const textMatchConfidence = document.getElementById('text-match-confidence');
    const textMatchMargin = document.getElementById('text-match-margin');
    const guessRankingsList = document.getElementById('guess-rankings-list');
    const btnGuessCorrect = document.getElementById('btn-guess-correct');
    const btnGuessWrong = document.getElementById('btn-guess-wrong');

    // Arena: Impostor Challenge
    const selectImpostorTarget = document.getElementById('select-impostor-target');
    const btnRecordMimic = document.getElementById('btn-record-mimic');
    const textRecordMimic = document.getElementById('text-record-mimic');
    const canvasMimic = document.getElementById('canvas-mimic-waveform');
    const cardMimicResult = document.getElementById('card-mimic-result');
    const textMimicVerdict = document.getElementById('text-mimic-verdict');
    const textMimicPoints = document.getElementById('text-mimic-points');
    const textMimicScore = document.getElementById('text-mimic-score');
    const barMimicProgress = document.getElementById('bar-mimic-progress');

    // Modals
    const modalGuide = document.getElementById('modal-guide');
    const btnCloseGuide = document.getElementById('btn-close-guide');
    const btnGuideGotIt = document.getElementById('btn-guide-got-it');

    const modalLeaderboard = document.getElementById('modal-leaderboard');
    const btnCloseLeaderboard = document.getElementById('btn-close-leaderboard');
    const leaderboardList = document.getElementById('leaderboard-list');

    const modalWrongSpeaker = document.getElementById('modal-wrong-speaker');
    const btnCloseWrongSpeaker = document.getElementById('btn-close-wrong-speaker');
    const speakerSelectList = document.getElementById('speaker-select-list');

    const modalReset = document.getElementById('modal-reset');
    const btnCloseReset = document.getElementById('btn-close-reset');
    const btnConfirmResetScores = document.getElementById('btn-confirm-reset-scores');
    const btnConfirmResetAll = document.getElementById('btn-confirm-reset-all');

    const toastContainer = document.getElementById('toast-container');

    // -------------------------------------------------------------
    // Toast Notification
    // -------------------------------------------------------------
    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerText = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.2s ease';
            setTimeout(() => toast.remove(), 200);
        }, 2800);
    }

    // -------------------------------------------------------------
    // Sound FX Toggle
    // -------------------------------------------------------------
    function updateSoundBtn() {
        if (sounds.isEnabled()) {
            btnToggleSound.classList.add('active');
        } else {
            btnToggleSound.classList.remove('active');
        }
    }

    btnToggleSound.addEventListener('click', () => {
        const on = sounds.toggle();
        updateSoundBtn();
        showToast(on ? 'Sound on' : 'Sound muted');
    });
    updateSoundBtn();

    // -------------------------------------------------------------
    // Segmented Mode Switching
    // -------------------------------------------------------------
    function switchMode(mode) {
        sounds.click();
        state.activeMode = mode;
        if (mode === 'guess_who') {
            tabGuessWho.classList.add('active');
            tabImpostor.classList.remove('active');
            sectionGuessWho.classList.remove('hidden');
            sectionImpostor.classList.add('hidden');
        } else {
            tabImpostor.classList.add('active');
            tabGuessWho.classList.remove('active');
            sectionImpostor.classList.remove('hidden');
            sectionGuessWho.classList.add('hidden');
            updateTargetDropdown();
        }
    }

    tabGuessWho.addEventListener('click', () => switchMode('guess_who'));
    tabImpostor.addEventListener('click', () => switchMode('impostor'));

    // -------------------------------------------------------------
    // Suggested Speech Phrase Shuffle
    // -------------------------------------------------------------
    btnShufflePhrase.addEventListener('click', () => {
        sounds.click();
        phraseIndex = (phraseIndex + 1) % SUGGESTED_PHRASES.length;
        textSuggestedPhrase.innerText = `"${SUGGESTED_PHRASES[phraseIndex]}"`;
    });

    // -------------------------------------------------------------
    // Local Roster Management
    // -------------------------------------------------------------
    function refreshRoster() {
        state.players = LocalEngine.LocalRoster.getPlayers();
        renderRoster();
        updateTargetDropdown();
        enrolledCountEl.innerText = state.players.length === 1 ? '1 Player' : `${state.players.length} Players`;
    }

    function renderRoster() {
        rosterChipsEl.innerHTML = '';
        if (state.players.length === 0) {
            rosterEmptyEl.classList.remove('hidden');
            rosterChipsEl.classList.add('hidden');
            return;
        }

        rosterEmptyEl.classList.add('hidden');
        rosterChipsEl.classList.remove('hidden');

        state.players.forEach((p) => {
            const chip = document.createElement('div');
            chip.className = 'player-chip';

            const name = document.createElement('span');
            name.className = 'player-chip-name';
            name.innerText = p.name;

            const score = document.createElement('span');
            score.className = 'player-chip-score';
            score.innerText = `${p.score} pts`;

            const del = document.createElement('button');
            del.className = 'player-chip-del';
            del.title = `Delete ${p.name}`;
            del.innerHTML = '&times;';
            del.addEventListener('click', (e) => {
                e.stopPropagation();
                sounds.click();
                LocalEngine.LocalRoster.removePlayer(p.player_id);
                showToast(`Removed '${p.name}'`);
                refreshRoster();
            });

            chip.appendChild(name);
            chip.appendChild(score);
            chip.appendChild(del);
            rosterChipsEl.appendChild(chip);
        });
    }

    function updateTargetDropdown() {
        const val = selectImpostorTarget.value;
        selectImpostorTarget.innerHTML = '<option value="">Select target player...</option>';
        state.players.forEach((p) => {
            const opt = document.createElement('option');
            opt.value = p.player_id;
            opt.innerText = p.name;
            selectImpostorTarget.appendChild(opt);
        });
        if (val && state.players.some((p) => p.player_id === val)) {
            selectImpostorTarget.value = val;
        }
    }

    // -------------------------------------------------------------
    // Recording: Player Registration
    // -------------------------------------------------------------
    let regTimer = null;

    btnRecordReg.addEventListener('click', async () => {
        const name = inputPlayerName.value.trim();
        if (!name) {
            showToast('Enter player name first');
            inputPlayerName.focus();
            return;
        }

        if (state.isRecording) {
            await finishRegistration(name);
            return;
        }

        try {
            await recorder.start();
            sounds.recordStart();
            state.isRecording = true;
            state.recordingType = 'reg';

            btnRecordReg.classList.add('recording');
            textRecordReg.innerText = 'Recording (10s)...';

            let timeLeft = 10;
            regTimer = setInterval(async () => {
                timeLeft--;
                if (timeLeft > 0) {
                    sounds.tick();
                    textRecordReg.innerText = `Recording (${timeLeft}s)...`;
                } else {
                    clearInterval(regTimer);
                    await finishRegistration(name);
                }
            }, 1000);
        } catch (err) {
            showToast('Microphone access denied');
        }
    });

    async function finishRegistration(name) {
        if (regTimer) {
            clearInterval(regTimer);
            regTimer = null;
        }

        sounds.recordStop();
        textRecordReg.innerText = 'Processing...';
        btnRecordReg.classList.remove('recording');

        try {
            const blob = await recorder.stop();
            state.isRecording = false;
            state.recordingType = null;
            if (!blob) return;

            // Extract biometric embedding vector from server
            const res = await api.extractEmbedding(blob);
            // Save player locally in browser localStorage
            const player = LocalEngine.LocalRoster.addPlayer(name, res.embedding);

            sounds.successChime();
            showToast(`'${player.name}' registered!`);
            inputPlayerName.value = '';
            refreshRoster();
        } catch (err) {
            sounds.failureTone();
            showToast(err.message || 'Registration failed');
        } finally {
            textRecordReg.innerText = 'Record Sample';
        }
    }

    // -------------------------------------------------------------
    // Recording: Mode A (Guess Who)
    // -------------------------------------------------------------
    btnRecordGuess.addEventListener('click', async () => {
        if (state.players.length === 0) {
            showToast('Register at least 1 player first');
            return;
        }

        if (state.isRecording) {
            await finishGuessWho();
            return;
        }

        try {
            await recorder.start(canvasGuess);
            sounds.recordStart();
            state.isRecording = true;
            state.recordingType = 'guess';

            cardGuessResult.classList.add('hidden');
            btnRecordGuess.classList.add('recording');
            textRecordGuess.innerText = 'Listening... (Tap to stop)';
        } catch (err) {
            showToast('Microphone access denied');
        }
    });

    async function finishGuessWho() {
        sounds.recordStop();
        textRecordGuess.innerText = 'Analyzing...';
        btnRecordGuess.classList.remove('recording');

        try {
            const blob = await recorder.stop();
            state.isRecording = false;
            state.recordingType = null;
            if (!blob) return;

            // Extract embedding vector from audio
            const audioData = await api.extractEmbedding(blob);
            // Run client-side 1-of-N cosine similarity matching
            const match = LocalEngine.identifySpeaker(audioData.embedding, state.players, 0.65);

            state.currentRound = {
                embedding: audioData.embedding,
                match: match,
            };

            textPredictedName.innerText = match.winner_name || 'No Match';
            const confDisplay = match.top_probability_percent !== undefined
                ? `${match.top_probability_percent}% Match`
                : `${match.confidence_percent}% Match`;
            textMatchConfidence.innerText = confDisplay;
            textMatchMargin.innerText = match.is_certain ? 'High Margin' : (match.margin > 0.15 ? 'Moderate Margin' : 'Disputed');

            // Candidate breakdown
            guessRankingsList.innerHTML = '';
            if (match.rankings && match.rankings.length > 0) {
                match.rankings.forEach((cand) => {
                    const row = document.createElement('div');
                    row.className = 'ranking-row';

                    const name = document.createElement('span');
                    name.innerText = cand.name;

                    const pct = document.createElement('span');
                    pct.style.fontWeight = '600';
                    pct.style.color = cand.player_id === match.winner_player_id ? 'var(--accent)' : 'var(--text-muted)';
                    pct.innerText = cand.probability_percent !== undefined
                        ? `${cand.similarity_percent}% (${cand.probability_percent}% prob)`
                        : `${cand.similarity_percent}%`;

                    row.appendChild(name);
                    row.appendChild(pct);
                    guessRankingsList.appendChild(row);
                });
            }

            if (match.confidence_percent >= 65) {
                sounds.successChime();
            } else {
                sounds.failureTone();
            }

            cardGuessResult.classList.remove('hidden');
        } catch (err) {
            sounds.failureTone();
            showToast(err.message || 'Analysis failed');
        } finally {
            textRecordGuess.innerText = 'Tap to Speak';
        }
    }

    btnGuessCorrect.addEventListener('click', () => {
        if (!state.currentRound || !state.currentRound.match || !state.currentRound.match.winner_player_id) return;
        sounds.click();

        const match = state.currentRound.match;
        const updated = LocalEngine.LocalRoster.claimRound(match.winner_player_id, state.currentRound.embedding, 50);

        if (updated) {
            sounds.successChime();
            showToast(`Correct! +50 pts to ${updated.name}`);
        }
        cardGuessResult.classList.add('hidden');
        state.currentRound = null;
        refreshRoster();
    });

    btnGuessWrong.addEventListener('click', () => {
        if (!state.currentRound) return;
        sounds.click();
        renderSpeakerSelectList();
        modalWrongSpeaker.classList.remove('hidden');
    });

    function renderSpeakerSelectList() {
        speakerSelectList.innerHTML = '';
        state.players.forEach((p) => {
            const row = document.createElement('button');
            row.className = 'btn-secondary';
            row.style.textAlign = 'left';
            row.style.display = 'flex';
            row.style.justifyContent = 'space-between';
            row.innerHTML = `<span>${p.name}</span><span style="color: var(--text-muted); font-size: 0.75rem;">${p.score} pts</span>`;

            row.addEventListener('click', () => {
                sounds.click();
                const updated = LocalEngine.LocalRoster.claimRound(p.player_id, state.currentRound.embedding, 50);
                if (updated) {
                    sounds.successChime();
                    showToast(`Claimed! +50 pts to ${updated.name}`);
                }
                modalWrongSpeaker.classList.add('hidden');
                cardGuessResult.classList.add('hidden');
                state.currentRound = null;
                refreshRoster();
            });

            speakerSelectList.appendChild(row);
        });
    }

    btnCloseWrongSpeaker.addEventListener('click', () => {
        sounds.click();
        modalWrongSpeaker.classList.add('hidden');
    });

    // -------------------------------------------------------------
    // Recording: Mode B (Impostor Challenge)
    // -------------------------------------------------------------
    btnRecordMimic.addEventListener('click', async () => {
        const targetId = selectImpostorTarget.value;
        if (!targetId) {
            showToast('Select target player first');
            selectImpostorTarget.focus();
            return;
        }

        if (state.isRecording) {
            await finishMimic(targetId);
            return;
        }

        try {
            await recorder.start(canvasMimic);
            sounds.recordStart();
            state.isRecording = true;
            state.recordingType = 'mimic';

            cardMimicResult.classList.add('hidden');
            btnRecordMimic.classList.add('recording');
            textRecordMimic.innerText = 'Mimicking... (Tap to stop)';
        } catch (err) {
            showToast('Microphone access denied');
        }
    });

    async function finishMimic(targetId) {
        sounds.recordStop();
        textRecordMimic.innerText = 'Evaluating...';
        btnRecordMimic.classList.remove('recording');

        try {
            const blob = await recorder.stop();
            state.isRecording = false;
            state.recordingType = null;
            if (!blob) return;

            const targetPlayer = state.players.find((p) => p.player_id === targetId);
            if (!targetPlayer) {
                showToast('Target player not found');
                return;
            }

            const audioData = await api.extractEmbedding(blob);
            const evalResult = LocalEngine.evaluateImpostor(audioData.embedding, targetPlayer, 0.60, 0.82);

            textMimicVerdict.innerText = evalResult.message;
            textMimicScore.innerText = `${evalResult.similarity_percent}%`;
            barMimicProgress.style.width = `${Math.min(100, Math.max(0, evalResult.similarity_percent))}%`;
            textMimicPoints.innerText = `+${evalResult.points} pts`;

            if (evalResult.points > 0) {
                LocalEngine.LocalRoster.addScore(targetPlayer.player_id, evalResult.points);
            }

            if (evalResult.security_breached || evalResult.status === 'CLOSE_MIMIC') {
                sounds.successChime();
            } else {
                sounds.failureTone();
            }

            cardMimicResult.classList.remove('hidden');
            refreshRoster();
        } catch (err) {
            sounds.failureTone();
            showToast(err.message || 'Mimic evaluation failed');
        } finally {
            textRecordMimic.innerText = 'Attempt Mimicry';
        }
    }

    // -------------------------------------------------------------
    // Modals: Guide, Leaderboard, Reset
    // -------------------------------------------------------------
    btnOpenGuide.addEventListener('click', () => {
        sounds.click();
        modalGuide.classList.remove('hidden');
    });

    btnCloseGuide.addEventListener('click', () => {
        sounds.click();
        modalGuide.classList.add('hidden');
    });

    btnGuideGotIt.addEventListener('click', () => {
        sounds.click();
        modalGuide.classList.add('hidden');
    });

    btnOpenLeaderboard.addEventListener('click', () => {
        sounds.click();
        const players = LocalEngine.LocalRoster.getScoreboard();
        renderLeaderboard(players);
        modalLeaderboard.classList.remove('hidden');
    });

    btnCloseLeaderboard.addEventListener('click', () => {
        sounds.click();
        modalLeaderboard.classList.add('hidden');
    });

    function renderLeaderboard(players) {
        leaderboardList.innerHTML = '';
        if (players.length === 0) {
            leaderboardList.innerHTML = '<p style="text-align: center; color: var(--text-muted); font-size: 0.8125rem;">No scores yet.</p>';
            return;
        }

        players.forEach((p, idx) => {
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.justifyContent = 'space-between';
            row.style.padding = '0.4rem 0.6rem';
            row.style.background = 'var(--bg-subtle)';
            row.style.borderRadius = 'var(--radius-sm)';
            row.style.fontSize = '0.8125rem';

            row.innerHTML = `
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <span style="color: var(--text-muted); font-weight: 600;">#${idx + 1}</span>
                    <span style="font-weight: 600;">${p.name}</span>
                </div>
                <span style="font-weight: 700; color: var(--accent);">${p.score} pts</span>
            `;
            leaderboardList.appendChild(row);
        });
    }

    btnOpenReset.addEventListener('click', () => {
        sounds.click();
        modalReset.classList.remove('hidden');
    });

    btnCloseReset.addEventListener('click', () => {
        sounds.click();
        modalReset.classList.add('hidden');
    });

    btnConfirmResetScores.addEventListener('click', () => {
        sounds.click();
        LocalEngine.LocalRoster.resetScoresOnly();
        showToast('Scores reset to zero');
        modalReset.classList.add('hidden');
        refreshRoster();
    });

    btnConfirmResetAll.addEventListener('click', () => {
        sounds.click();
        LocalEngine.LocalRoster.clearAll();
        showToast('All cleared');
        modalReset.classList.add('hidden');
        cardGuessResult.classList.add('hidden');
        cardMimicResult.classList.add('hidden');
        refreshRoster();
    });

    // Esc closes modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            modalGuide.classList.add('hidden');
            modalLeaderboard.classList.add('hidden');
            modalWrongSpeaker.classList.add('hidden');
            modalReset.classList.add('hidden');
        }
    });

    // Initial load from device localStorage
    refreshRoster();
});
