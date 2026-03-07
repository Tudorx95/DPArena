// OutputCell.jsx (enhanced with real-time progress messages)
import React from 'react';
import { Loader2, AlertCircle, CheckCircle, XCircle, TrendingUp } from 'lucide-react';
import ProgressStep from './ProgressStep';
import ExportPDFButton from './ExportPDFButton';

export default function OutputCell({ output, isLoading, orchestratorStatus, onCancel, isCancelling, fileName }) {
    // Define all simulation steps with their metadata
    const steps = [
        { id: 1, name: 'Preparing Environment', key: 'initialization' },
        { id: 2, name: 'Code Execution', key: 'code_execution' },
        { id: 3, name: 'Data Generation', key: 'data_generation' },
        { id: 4, name: 'Data Poisoning', key: 'data_poisoning' },
        { id: 5, name: 'FL Simulation (Clean)', key: 'fl_simulation_clean' },
        { id: 6, name: 'FL Simulation (Clean Data Protection)', key: 'fl_simulation_clean_dp' },
        { id: 7, name: 'FL Simulation (Poisoned)', key: 'fl_simulation_poisoned' },
        { id: 8, name: 'FL Simulation (Data Poison Protection)', key: 'fl_simulation_poisoned_dp' },
        { id: 9, name: 'Generating Results', key: 'generate_results' }
    ];

    // Debug logging to see what data we're receiving
    React.useEffect(() => {
        if (orchestratorStatus) {
            console.log('🎨 OutputCell - orchestratorStatus:', {
                status: orchestratorStatus.status,
                step: orchestratorStatus.step,
                message: orchestratorStatus.message,
                timestamp: orchestratorStatus.timestamp,
                fullData: orchestratorStatus
            });
        }
    }, [orchestratorStatus]);

    // Get the current step number from orchestrator
    const getCurrentStep = () => {
        // Orchestrator sends step directly (1-7)
        if (orchestratorStatus?.step) {
            return orchestratorStatus.step;
        }
        return 1;
    };

    // Determine status for each step
    const getStepStatus = (stepId) => {
        const currentStep = getCurrentStep();
        const overallStatus = orchestratorStatus?.status;

        if (overallStatus === 'error') {
            // If there's an error, mark current step as error, previous as completed
            if (stepId < currentStep) return 'completed';
            if (stepId === currentStep) return 'error';
            return 'pending';
        }

        if (overallStatus === 'completed') {
            return 'completed';
        }

        if (overallStatus === 'cancelled') {
            if (stepId < currentStep) return 'completed';
            if (stepId === currentStep) return 'cancelled';
            return 'pending';
        }

        // Normal running state
        if (stepId < currentStep) return 'completed';
        if (stepId === currentStep) return 'running';
        return 'pending';
    };

    // Get the message for the current active step
    const getStepMessage = (stepId) => {
        const currentStep = getCurrentStep();

        // Only show message for the currently active step
        if (stepId === currentStep && orchestratorStatus?.message) {
            return orchestratorStatus.message;
        }

        return null;
    };

    // Get timestamp for step
    const getStepTimestamp = (stepId) => {
        const currentStep = getCurrentStep();
        if (stepId === currentStep && orchestratorStatus?.timestamp) {
            return orchestratorStatus.timestamp;
        }
        return null;
    };

    // Calculate overall progress percentage
    const getProgressPercentage = () => {
        if (orchestratorStatus?.status === 'completed') return 100;
        if (orchestratorStatus?.status === 'error') return 0;
        if (orchestratorStatus?.status === 'cancelled') {
            const currentStep = getCurrentStep();
            return Math.round((currentStep / steps.length) * 100);
        }

        const currentStep = getCurrentStep();
        // Each step represents roughly equal progress
        return Math.round((currentStep / steps.length) * 100);
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 overflow-hidden flex flex-col h-full">
            <div className="p-3 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600 flex items-center justify-between flex-shrink-0">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-600 dark:text-gray-300">Output</span>
                    {(isLoading || orchestratorStatus?.status === 'running') && (
                        <span className="flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 px-2 py-1 rounded-full">
                            <TrendingUp className="w-3 h-3" />
                            {getProgressPercentage()}% Complete
                        </span>
                    )}
                    {orchestratorStatus?.status === 'completed' && (
                        <span className="flex items-center gap-1 text-xs text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/30 px-2 py-1 rounded-full">
                            <CheckCircle className="w-3 h-3" />
                            Completed
                        </span>
                    )}
                </div>

                {/* Cancel Button - only show when simulation is running */}
                {(isLoading || orchestratorStatus?.status === 'running') && (
                    <button
                        onClick={onCancel}
                        disabled={isCancelling}
                        className="flex items-center gap-2 px-3 py-1.5 bg-red-600 text-white text-sm rounded hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                    >
                        {isCancelling ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Cancelling...
                            </>
                        ) : (
                            <>
                                <XCircle className="w-4 h-4" />
                                Cancel Simulation
                            </>
                        )}
                    </button>
                )}

                {/* Export PDF Button - only show when simulation is completed */}
                {orchestratorStatus?.status === 'completed' && orchestratorStatus?.results_data && (
                    <ExportPDFButton
                        results={orchestratorStatus.results_data}
                        fileName={fileName || 'simulation'}
                    />
                )}
            </div>

            <div className="flex-1 p-4 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 font-mono text-sm overflow-auto sidebar-scroll">
                {isLoading || orchestratorStatus?.status === 'running' ? (
                    <div className="space-y-4">
                        {/* Main Status Indicator */}
                        <div className="flex items-center justify-center mb-6 p-4 bg-blue-50 dark:bg-gray-800 rounded-lg border border-blue-200 dark:border-gray-700">
                            <Loader2 className="w-8 h-8 animate-spin text-blue-600 dark:text-blue-400 mr-3" />
                            <div>
                                <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                                    Simulation in Progress
                                </p>
                                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                                    Step {getCurrentStep()} of {steps.length}
                                    {orchestratorStatus?.message && ` - ${orchestratorStatus.message}`}
                                </p>
                            </div>
                        </div>

                        {/* Progress Steps with Real-time Messages */}
                        <div className="space-y-2">
                            {steps.map(step => (
                                <ProgressStep
                                    key={step.id}
                                    step={step.id}
                                    stepName={step.name}
                                    currentStep={getCurrentStep()}
                                    status={getStepStatus(step.id)}
                                    message={getStepMessage(step.id)}
                                    timestamp={getStepTimestamp(step.id)}
                                />
                            ))}
                        </div>

                        {/* Overall Progress Bar */}
                        <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-gray-600 dark:text-gray-400 font-semibold">
                                    Overall Progress
                                </span>
                                <span className="text-xs text-blue-600 dark:text-blue-400 font-bold">
                                    {getProgressPercentage()}%
                                </span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                                <div
                                    className="bg-gradient-to-r from-blue-500 to-blue-400 h-3 rounded-full transition-all duration-500 ease-out"
                                    style={{ width: `${getProgressPercentage()}%` }}
                                />
                            </div>
                        </div>

                        {/* Live Console Output Section */}
                        {output && (
                            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                                <p className="text-xs text-gray-600 dark:text-gray-400 mb-2 font-semibold">Console Output:</p>
                                <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{output}</pre>
                            </div>
                        )}
                    </div>
                ) : orchestratorStatus?.status === 'cancelled' ? (
                    // Cancellation Status Display
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 p-4 bg-orange-50 dark:bg-orange-900/30 border border-orange-300 dark:border-orange-700 rounded-lg">
                            <XCircle className="w-8 h-8 text-orange-600 dark:text-orange-400 flex-shrink-0" />
                            <div>
                                <h3 className="text-lg font-semibold text-orange-600 dark:text-orange-400">
                                    Simulation Cancelled
                                </h3>
                                <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                                    The simulation was cancelled by user request at step {getCurrentStep()} of {steps.length}.
                                </p>
                            </div>
                        </div>

                        {/* Show completed steps */}
                        <div className="space-y-2">
                            <p className="text-sm text-gray-600 dark:text-gray-400 font-semibold mb-2">Steps Completed Before Cancellation:</p>
                            {steps.filter(step => step.id < getCurrentStep()).map(step => (
                                <ProgressStep
                                    key={step.id}
                                    step={step.id}
                                    stepName={step.name}
                                    currentStep={getCurrentStep()}
                                    status="completed"
                                    message={null}
                                    timestamp={null}
                                />
                            ))}
                        </div>

                        {orchestratorStatus.message && (
                            <div className="p-3 bg-orange-50 dark:bg-gray-800 rounded-lg border border-orange-300 dark:border-orange-700/50">
                                <p className="text-xs text-orange-600 dark:text-orange-400 font-semibold mb-1">Cancellation Details:</p>
                                <p className="text-sm text-gray-700 dark:text-gray-300">{orchestratorStatus.message}</p>
                            </div>
                        )}
                    </div>
                ) : orchestratorStatus?.status === 'error' ? (
                    // Error Status Display
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 p-4 bg-red-50 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg">
                            <AlertCircle className="w-8 h-8 text-red-600 dark:text-red-400 flex-shrink-0" />
                            <div>
                                <h3 className="text-lg font-semibold text-red-600 dark:text-red-400">Error Occurred</h3>
                                <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                                    The simulation encountered an error at step {getCurrentStep()}.
                                </p>
                            </div>
                        </div>

                        {/* Error Message */}
                        {orchestratorStatus.message && (
                            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700/50 rounded-lg">
                                <p className="text-xs text-red-600 dark:text-red-400 font-semibold mb-2">Error Details:</p>
                                <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                                    {orchestratorStatus.message}
                                </pre>
                            </div>
                        )}

                        {/* Show steps that completed before error */}
                        <div className="space-y-2">
                            <p className="text-sm text-gray-600 dark:text-gray-400 font-semibold mb-2">Progress Before Error:</p>
                            {steps.map(step => (
                                <ProgressStep
                                    key={step.id}
                                    step={step.id}
                                    stepName={step.name}
                                    currentStep={getCurrentStep()}
                                    status={getStepStatus(step.id)}
                                    message={step.id === getCurrentStep() ? orchestratorStatus.message : null}
                                    timestamp={null}
                                />
                            ))}
                        </div>
                    </div>
                ) : orchestratorStatus?.status === 'completed' ? (
                    // Completion Status Display
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/30 border border-green-300 dark:border-green-700 rounded-lg">
                            <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400 flex-shrink-0" />
                            <div>
                                <h3 className="text-lg font-semibold text-green-600 dark:text-green-400">
                                    Simulation Completed Successfully!
                                </h3>
                                <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                                    All {steps.length} steps completed without errors.
                                </p>
                            </div>
                        </div>

                        {/* Show all completed steps */}
                        <div className="space-y-2">
                            <p className="text-sm text-green-600 dark:text-green-400 font-semibold mb-2">Pipeline Summary:</p>
                            {steps.map(step => (
                                <ProgressStep
                                    key={step.id}
                                    step={step.id}
                                    stepName={step.name}
                                    currentStep={steps.length + 1}
                                    status="completed"
                                    message={null}
                                    timestamp={null}
                                />
                            ))}
                        </div>

                        {/* Simulation Config Info (filtered console output) */}
                        {output && (() => {
                            const filteredLines = output
                                .split('\n')
                                .filter(line => {
                                    const trimmed = line.trim().toLowerCase();
                                    return !trimmed.includes('sending simulation to orchestrator') &&
                                        !trimmed.includes('simulation queued on orchestrator') &&
                                        !trimmed.includes('✅');
                                })
                                .join('\n')
                                .trim();

                            return filteredLines ? (
                                <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-700/50">
                                    <p className="text-xs text-indigo-600 dark:text-indigo-400 font-semibold mb-2">Simulation Details:</p>
                                    <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{filteredLines}</pre>
                                </div>
                            ) : null;
                        })()}

                        {/* Final Results */}
                        {orchestratorStatus.results_data && (
                            <div className="mt-4 p-4 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/30 dark:to-purple-900/30 border border-blue-300 dark:border-blue-700 rounded-lg">
                                <p className="text-blue-600 dark:text-blue-400 mb-3 font-semibold flex items-center gap-2">
                                    <TrendingUp className="w-4 h-4" />
                                    Final Results
                                </p>
                                <pre className="text-gray-700 dark:text-gray-300 text-xs p-3 bg-gray-100 dark:bg-gray-800/50 rounded overflow-x-auto">
                                    {orchestratorStatus.results_data.summary}
                                </pre>
                                <details className="mt-3">
                                    <summary className="text-xs text-blue-600 dark:text-blue-400 cursor-pointer hover:text-blue-500 dark:hover:text-blue-300 transition-colors">
                                        View detailed analysis (JSON) →
                                    </summary>
                                    <pre className="text-gray-700 dark:text-gray-300 text-xs mt-2 p-3 bg-gray-100 dark:bg-gray-800/50 rounded overflow-x-auto">
                                        {JSON.stringify(orchestratorStatus.results_data.analysis, null, 2)}
                                    </pre>
                                </details>
                            </div>
                        )}
                    </div>
                ) : (
                    // Default output display (no simulation running)
                    <pre className="whitespace-pre-wrap text-gray-700 dark:text-gray-300">{output || 'Ready to run simulation...'}</pre>
                )}
            </div>
        </div>
    );
}