// src/context/SimulationContext.jsx
import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';

const SimulationContext = createContext(null);

const STORAGE_KEYS = {
    SIMULATIONS: 'fl_simulations_history',
    ACTIVE_SIM: 'fl_active_simulation',
    CONFIG: 'fl_simulation_config',
    SIMULATION_OUTPUT: 'fl_simulation_output',
    FILE_SIM_STATES: 'fl_file_simulation_states',
    COMPLETED_SIMS: 'fl_completed_simulations',
    ACTIVE_PROJECT: 'fl_active_project_id',
    ACTIVE_FILE: 'fl_active_file_id'
};

export const SimulationProvider = ({ children }) => {
    // Încarcă simulările din localStorage
    const [simulations, setSimulations] = useState(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.SIMULATIONS);
            return saved ? JSON.parse(saved) : [];
        } catch (error) {
            console.error('Error loading simulations:', error);
            return [];
        }
    });

    // Simularea activă curentă
    const [activeSimulation, setActiveSimulation] = useState(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.ACTIVE_SIM);
            return saved ? JSON.parse(saved) : null;
        } catch (error) {
            console.error('Error loading active simulation:', error);
            return null;
        }
    });

    // Output-ul simulării (pentru OutputCell)
    const [simulationOutput, setSimulationOutput] = useState(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.SIMULATION_OUTPUT);
            return saved ? JSON.parse(saved) : null;
        } catch (error) {
            console.error('Error loading simulation output:', error);
            return null;
        }
    });

    // Valid poison operations (must match poison_data_v2.py)
    const VALID_POISON_OPERATIONS = [
        'label_flip', 'backdoor_badnets', 'backdoor_blended',
        'backdoor_sig', 'backdoor_trojan', 'semantic_backdoor', 'backdoor_edge_case'
    ];

    // Configurația curentă
    const [config, setConfig] = useState(() => {
        const defaultConfig = {
            N: 10,
            M: 2,
            NN_NAME: 'SimpleNN',
            R: 5,
            ROUNDS: 10,
            EPOCHS: 3,
            strategy: 'first',
            poison_operation: 'backdoor_blended',
            poison_intensity: 0.1,
            poison_percentage: 0.2,
            data_poison_protection: 'fedavg',
            // Data distribution parameters
            data_distribution: 'fixed',
            dominant_percentage: 80,
            dirichlet_alpha: 0.5,
            // Attack-specific sub-parameters
            target_class: '',
            no_flip: false,
            trigger_type: 'square',
            pattern_type: 'random',
            modification: 'green_tint',
            transform: 'rotation',
            watermark_type: 'apple'
        };
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.CONFIG);
            if (saved) {
                const parsed = JSON.parse(saved);
                // Migrate stale poison_operation values (e.g. 'noise')
                if (!VALID_POISON_OPERATIONS.includes(parsed.poison_operation)) {
                    parsed.poison_operation = defaultConfig.poison_operation;
                }
                // Merge defaults so newly-added fields are always present
                return { ...defaultConfig, ...parsed };
            }
            return defaultConfig;
        } catch (error) {
            console.error('Error loading config:', error);
            return defaultConfig;
        }
    });

    const [loading, setLoading] = useState(false);

    // State-uri adiționale pentru persistență
    const [fileSimulationStates, setFileSimulationStates] = useState(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.FILE_SIM_STATES);
            return saved ? JSON.parse(saved) : {};
        } catch (error) {
            console.error('Error loading file simulation states:', error);
            return {};
        }
    });

    const [completedSimulations, setCompletedSimulations] = useState(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.COMPLETED_SIMS);
            return saved ? JSON.parse(saved) : {};
        } catch (error) {
            console.error('Error loading completed simulations:', error);
            return {};
        }
    });

    const [activeProjectId, setActiveProjectId] = useState(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.ACTIVE_PROJECT);
            return saved ? JSON.parse(saved) : null;
        } catch (error) {
            console.error('Error loading active project:', error);
            return null;
        }
    });

    const [activeFileId, setActiveFileId] = useState(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEYS.ACTIVE_FILE);
            return saved ? JSON.parse(saved) : null;
        } catch (error) {
            console.error('Error loading active file:', error);
            return null;
        }
    });

    // Salvează automat în localStorage
    useEffect(() => {
        try {
            localStorage.setItem(STORAGE_KEYS.SIMULATIONS, JSON.stringify(simulations));
        } catch (error) {
            console.error('Error saving simulations:', error);
        }
    }, [simulations]);

    useEffect(() => {
        try {
            if (activeSimulation) {
                localStorage.setItem(STORAGE_KEYS.ACTIVE_SIM, JSON.stringify(activeSimulation));
            } else {
                localStorage.removeItem(STORAGE_KEYS.ACTIVE_SIM);
            }
        } catch (error) {
            console.error('Error saving active simulation:', error);
        }
    }, [activeSimulation]);

    useEffect(() => {
        try {
            if (simulationOutput) {
                localStorage.setItem(STORAGE_KEYS.SIMULATION_OUTPUT, JSON.stringify(simulationOutput));
            } else {
                localStorage.removeItem(STORAGE_KEYS.SIMULATION_OUTPUT);
            }
        } catch (error) {
            console.error('Error saving simulation output:', error);
        }
    }, [simulationOutput]);

    useEffect(() => {
        try {
            localStorage.setItem(STORAGE_KEYS.CONFIG, JSON.stringify(config));
        } catch (error) {
            console.error('Error saving config:', error);
        }
    }, [config]);

    // Salvează state-urile adiționale
    useEffect(() => {
        try {
            localStorage.setItem(STORAGE_KEYS.FILE_SIM_STATES, JSON.stringify(fileSimulationStates));
        } catch (error) {
            console.error('Error saving file simulation states:', error);
        }
    }, [fileSimulationStates]);

    useEffect(() => {
        try {
            localStorage.setItem(STORAGE_KEYS.COMPLETED_SIMS, JSON.stringify(completedSimulations));
        } catch (error) {
            console.error('Error saving completed simulations:', error);
        }
    }, [completedSimulations]);

    useEffect(() => {
        try {
            if (activeProjectId) {
                localStorage.setItem(STORAGE_KEYS.ACTIVE_PROJECT, JSON.stringify(activeProjectId));
            } else {
                localStorage.removeItem(STORAGE_KEYS.ACTIVE_PROJECT);
            }
        } catch (error) {
            console.error('Error saving active project:', error);
        }
    }, [activeProjectId]);

    useEffect(() => {
        try {
            if (activeFileId) {
                localStorage.setItem(STORAGE_KEYS.ACTIVE_FILE, JSON.stringify(activeFileId));
            } else {
                localStorage.removeItem(STORAGE_KEYS.ACTIVE_FILE);
            }
        } catch (error) {
            console.error('Error saving active file:', error);
        }
    }, [activeFileId]);

    // Pornește o simulare nouă
    const startSimulation = useCallback((fileId, projectId) => {
        const newSim = {
            id: `sim_${Date.now()}`,
            fileId,
            projectId,
            config: { ...config },
            status: 'running',
            startTime: new Date().toISOString(),
            progress: 0,
            currentStep: 1,
            steps: {
                1: { name: 'Initializing FL Environment', status: 'running', message: 'Setting up...', timestamp: new Date().toISOString() },
                2: { name: 'Distributing Data to Clients', status: 'pending', message: null, timestamp: null },
                3: { name: 'Training Rounds', status: 'pending', message: null, timestamp: null },
                4: { name: 'Aggregating Models', status: 'pending', message: null, timestamp: null },
                5: { name: 'Final Evaluation', status: 'pending', message: null, timestamp: null }
            },
            results: null,
            error: null
        };

        setActiveSimulation(newSim);
        setSimulations(prev => [newSim, ...prev]);
        setSimulationOutput({
            fileId,
            projectId,
            simulationId: newSim.id,
            isRunning: true,
            isCompleted: false
        });
        setLoading(true);

        return newSim.id;
    }, [config]);

    // Actualizează step-ul curent
    const updateSimulationStep = useCallback((id, stepNumber, updates) => {
        setSimulations(prev =>
            prev.map(sim =>
                sim.id === id
                    ? {
                        ...sim,
                        currentStep: stepNumber,
                        steps: {
                            ...sim.steps,
                            [stepNumber]: {
                                ...sim.steps[stepNumber],
                                ...updates,
                                timestamp: updates.timestamp || new Date().toISOString()
                            }
                        },
                        lastUpdated: new Date().toISOString()
                    }
                    : sim
            )
        );

        if (activeSimulation?.id === id) {
            setActiveSimulation(prev => ({
                ...prev,
                currentStep: stepNumber,
                steps: {
                    ...prev.steps,
                    [stepNumber]: {
                        ...prev.steps[stepNumber],
                        ...updates,
                        timestamp: updates.timestamp || new Date().toISOString()
                    }
                },
                lastUpdated: new Date().toISOString()
            }));
        }
    }, [activeSimulation]);

    // Actualizează progresul
    const updateProgress = useCallback((id, progress) => {
        setSimulations(prev =>
            prev.map(sim =>
                sim.id === id ? { ...sim, progress } : sim
            )
        );

        if (activeSimulation?.id === id) {
            setActiveSimulation(prev => ({ ...prev, progress }));
        }
    }, [activeSimulation]);

    // Finalizează simularea
    const completeSimulation = useCallback((id, results) => {
        const updates = {
            status: 'completed',
            endTime: new Date().toISOString(),
            progress: 100,
            results,
            error: null,
            currentStep: 5,
            steps: {
                ...activeSimulation?.steps,
                5: {
                    ...activeSimulation?.steps[5],
                    status: 'completed',
                    message: 'Simulation completed successfully',
                    timestamp: new Date().toISOString()
                }
            }
        };

        setSimulations(prev =>
            prev.map(sim => (sim.id === id ? { ...sim, ...updates } : sim))
        );

        if (activeSimulation?.id === id) {
            setActiveSimulation(prev => ({ ...prev, ...updates }));
        }

        setSimulationOutput(prev => ({
            ...prev,
            isRunning: false,
            isCompleted: true,
            results
        }));

        setLoading(false);

        // Oprește simularea activă după 3 secunde
        setTimeout(() => {
            setActiveSimulation(null);
        }, 3000);
    }, [activeSimulation]);

    // Marchează simularea ca eșuată
    const failSimulation = useCallback((id, errorMessage) => {
        const updates = {
            status: 'failed',
            endTime: new Date().toISOString(),
            error: errorMessage
        };

        setSimulations(prev =>
            prev.map(sim => (sim.id === id ? { ...sim, ...updates } : sim))
        );

        if (activeSimulation?.id === id) {
            setActiveSimulation(prev => ({ ...prev, ...updates }));
        }

        setSimulationOutput(prev => ({
            ...prev,
            isRunning: false,
            error: errorMessage
        }));

        setLoading(false);

        setTimeout(() => {
            setActiveSimulation(null);
        }, 2000);
    }, [activeSimulation]);

    // Oprește simularea
    const stopSimulation = useCallback((id) => {
        const updates = {
            status: 'stopped',
            endTime: new Date().toISOString()
        };

        setSimulations(prev =>
            prev.map(sim => (sim.id === id ? { ...sim, ...updates } : sim))
        );

        if (activeSimulation?.id === id) {
            setActiveSimulation(null);
        }

        setSimulationOutput(prev => ({
            ...prev,
            isRunning: false
        }));

        setLoading(false);
    }, [activeSimulation]);

    // Șterge output-ul simulării
    const clearSimulationOutput = useCallback(() => {
        setSimulationOutput(null);
        localStorage.removeItem(STORAGE_KEYS.SIMULATION_OUTPUT);
    }, []);

    // Șterge o simulare
    const deleteSimulation = useCallback((id) => {
        setSimulations(prev => prev.filter(sim => sim.id !== id));

        if (activeSimulation?.id === id) {
            setActiveSimulation(null);
            clearSimulationOutput();
        }
    }, [activeSimulation, clearSimulationOutput]);

    // Șterge toate simulările
    const clearAllSimulations = useCallback(() => {
        setSimulations([]);
        setActiveSimulation(null);
        clearSimulationOutput();
        localStorage.removeItem(STORAGE_KEYS.SIMULATIONS);
        localStorage.removeItem(STORAGE_KEYS.ACTIVE_SIM);
    }, [clearSimulationOutput]);

    const value = {
        // State
        simulations,
        activeSimulation,
        simulationOutput,
        config,
        loading,
        fileSimulationStates,
        completedSimulations,
        activeProjectId,
        activeFileId,

        // Actions
        setConfig,
        startSimulation,
        updateSimulationStep,
        updateProgress,
        completeSimulation,
        failSimulation,
        stopSimulation,
        clearSimulationOutput,
        deleteSimulation,
        clearAllSimulations,
        setFileSimulationStates,
        setCompletedSimulations,
        setActiveProjectId,
        setActiveFileId,

        // Computed
        hasActiveSimulation: !!activeSimulation,
        isRunning: activeSimulation?.status === 'running'
    };

    return (
        <SimulationContext.Provider value={value}>
            {children}
        </SimulationContext.Provider>
    );
};

export const useSimulation = () => {
    const context = useContext(SimulationContext);
    if (!context) {
        throw new Error('useSimulation must be used within SimulationProvider');
    }
    return context;
};