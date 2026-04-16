import React, { useState, useEffect } from 'react';
import { FileSpreadsheet, X, Loader2, CheckSquare, Square, Download } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function MultiExportCSV({ onClose, token, projects }) {
    const { isDarkMode } = useTheme();
    const [selectedFiles, setSelectedFiles] = useState(new Set());
    const [loading, setLoading] = useState(false);
    const [simulationData, setSimulationData] = useState({});
    const [loadingData, setLoadingData] = useState(false);
    const [expandedProjects, setExpandedProjects] = useState(new Set());

    const authHeaders = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };

    // Projects start collapsed — user expands manually

    // Handle ESC key to close modal
    useEffect(() => {
        const handleEscape = (event) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };

        document.addEventListener('keydown', handleEscape);
        return () => {
            document.removeEventListener('keydown', handleEscape);
        };
    }, [onClose]);

    // Load simulation data for selected files
    useEffect(() => {
        if (selectedFiles.size > 0) {
            loadSimulationDataForFiles();
        }
    }, [selectedFiles]);

    const loadSimulationDataForFiles = async () => {
        setLoadingData(true);
        const newData = { ...simulationData };

        for (const fileId of selectedFiles) {
            if (!newData[fileId]) {
                try {
                    const response = await fetch(`${API_URL}/api/files/${fileId}/simulation-results`, {
                        headers: authHeaders
                    });

                    if (response.ok) {
                        const result = await response.json();
                        if (result && result.results) {
                            newData[fileId] = result.results;
                        }
                    }
                } catch (error) {
                    console.error(`Failed to load simulation for file ${fileId}:`, error);
                }
            }
        }

        setSimulationData(newData);
        setLoadingData(false);
    };

    const toggleFileSelection = (fileId) => {
        const newSelected = new Set(selectedFiles);
        if (newSelected.has(fileId)) {
            newSelected.delete(fileId);
        } else {
            newSelected.add(fileId);
        }
        setSelectedFiles(newSelected);
    };

    const toggleProjectExpansion = (projectId) => {
        const newExpanded = new Set(expandedProjects);
        if (newExpanded.has(projectId)) {
            newExpanded.delete(projectId);
        } else {
            newExpanded.add(projectId);
        }
        setExpandedProjects(newExpanded);
    };

    const selectAllInProject = (project) => {
        const newSelected = new Set(selectedFiles);
        const filesWithSimulations = project.files.filter(file =>
            hasSimulationResults(file.id, project.id)
        );

        const allSelected = filesWithSimulations.every(file => newSelected.has(file.id));

        if (allSelected) {
            // Deselect all
            filesWithSimulations.forEach(file => newSelected.delete(file.id));
        } else {
            // Select all
            filesWithSimulations.forEach(file => newSelected.add(file.id));
        }

        setSelectedFiles(newSelected);
    };

    const hasSimulationResults = (fileId, projectId) => {
        // Check if file has simulation results (you might need to add this info to file metadata)
        // For now, we'll assume all files might have simulations
        return true;
    };

    const getFileInfo = (fileId) => {
        for (const project of projects) {
            const file = project.files.find(f => f.id === fileId);
            if (file) {
                return { file, project };
            }
        }
        return null;
    };

    const exportToCSV = async () => {
        if (selectedFiles.size === 0) {
            alert('Please select at least one simulation to export');
            return;
        }

        setLoading(true);
        try {
            // Prepare data rows
            const rows = [];

            // CSV Header (matching ExportPDFButton order)
            const headers = [
                'File Name',
                'Project Name',
                'Completed At',
                'Total Clients (N)',
                'Malicious Clients (M)',
                'Neural Network',
                'Training Rounds',
                'Poisoned Rounds (R)',
                'Distribution Strategy',
                'Poisoning Operation',
                'Attack Intensity (%)',
                'Poisoned Data Percentage (%)',
                'Init Accuracy',
                'Clean Accuracy',
                'Clean + DP Protection Accuracy',
                'Poisoned Accuracy',
                'Poisoned + DP Protection Accuracy',
                'Accuracy Drop (Clean - Poisoned)',
                'Drop (Clean - Init)',
                'Drop (Clean DP - Init)',
                'Drop (Poisoned - Init)',
                'Drop (Poisoned DP - Init)',
                'DP Protection Method',
                'GPU Used',
                'Clean Precision',
                'Clean Recall',
                'Clean F1',
                'Clean DP Precision',
                'Clean DP Recall',
                'Clean DP F1',
                'Poisoned Precision',
                'Poisoned Recall',
                'Poisoned F1',
                'DP Protection Precision',
                'DP Protection Recall',
                'DP Protection F1',
                'Summary'
            ];
            rows.push(headers);

            // Add data for each selected file
            for (const fileId of selectedFiles) {
                const info = getFileInfo(fileId);
                if (!info) continue;

                const { file, project } = info;
                const results = simulationData[fileId];

                if (!results) continue;

                const config = results.config || {};
                const analysis = results.analysis || {};

                // Map poisoning operation to readable text
                const operationMap = {
                    'noise': 'Gaussian Noise',
                    'label_flip': 'Label Flip',
                    'backdoor': 'Backdoor Trigger'
                };
                const operation = operationMap[config.poison_operation] || config.poison_operation || 'N/A';

                // Format percentages
                const attackIntensity = config.poison_intensity
                    ? (config.poison_intensity * 100).toFixed(1)
                    : 'N/A';
                const poisonedPercentage = config.poison_percentage
                    ? (config.poison_percentage * 100).toFixed(1)
                    : 'N/A';

                // Create row
                const row = [
                    file.name || 'N/A',
                    project.name || 'N/A',
                    results.completed_at ? new Date(results.completed_at).toLocaleString() : 'N/A',
                    config.N || 'N/A',
                    config.M || 'N/A',
                    config.NN_NAME || 'N/A',
                    config.ROUNDS || 'N/A',
                    config.R || 'N/A',
                    config.strategy || 'N/A',
                    operation,
                    attackIntensity,
                    poisonedPercentage,
                    analysis.init_accuracy?.toFixed(4) || 'N/A',
                    analysis.clean_accuracy?.toFixed(4) || 'N/A',
                    analysis.clean_dp_accuracy?.toFixed(4) || 'N/A',
                    analysis.poisoned_accuracy?.toFixed(4) || 'N/A',
                    analysis.poisoned_dp_accuracy?.toFixed(4) || 'N/A',
                    analysis.accuracy_drop?.toFixed(4) || 'N/A',
                    analysis.drop_clean_init?.toFixed(4) || 'N/A',
                    analysis.drop_clean_dp_init?.toFixed(4) || 'N/A',
                    analysis.drop_poison_init?.toFixed(4) || 'N/A',
                    analysis.drop_poison_dp_init?.toFixed(4) || 'N/A',
                    analysis.data_poison_protection_method || 'N/A',
                    analysis.gpu_used || 'N/A',
                    analysis.clean_precision?.toFixed(4) || 'N/A',
                    analysis.clean_recall?.toFixed(4) || 'N/A',
                    analysis.clean_f1?.toFixed(4) || 'N/A',
                    analysis.clean_dp_precision?.toFixed(4) || 'N/A',
                    analysis.clean_dp_recall?.toFixed(4) || 'N/A',
                    analysis.clean_dp_f1?.toFixed(4) || 'N/A',
                    analysis.poisoned_precision?.toFixed(4) || 'N/A',
                    analysis.poisoned_recall?.toFixed(4) || 'N/A',
                    analysis.poisoned_f1?.toFixed(4) || 'N/A',
                    analysis.poisoned_dp_precision?.toFixed(4) || 'N/A',
                    analysis.poisoned_dp_recall?.toFixed(4) || 'N/A',
                    analysis.poisoned_dp_f1?.toFixed(4) || 'N/A',
                    (results.summary || 'N/A').replace(/"/g, '""') // Escape quotes in summary
                ];

                rows.push(row);
            }

            // Convert to CSV string
            const csvContent = rows.map(row =>
                row.map(cell => {
                    // Wrap in quotes if contains comma, newline, or quote
                    const cellStr = String(cell);
                    if (cellStr.includes(',') || cellStr.includes('\n') || cellStr.includes('"')) {
                        return `"${cellStr}"`;
                    }
                    return cellStr;
                }).join(',')
            ).join('\n');

            // Create blob and download
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);

            link.setAttribute('href', url);
            link.setAttribute('download', `FL_Simulations_Export_${new Date().getTime()}.csv`);
            link.style.visibility = 'hidden';

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            alert(`Successfully exported ${selectedFiles.size} simulation(s) to CSV!`);
            onClose();

        } catch (error) {
            console.error('Export failed:', error);
            alert('Failed to export CSV. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const filesWithSimulations = selectedFiles.size;
    const filesLoaded = Object.keys(simulationData).length;

    const handleOverlayClick = (e) => {
        // Close modal when clicking on the overlay (not on the modal content)
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    return (
        <div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={handleOverlayClick}
        >
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                        <FileSpreadsheet className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                        <div>
                            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">Export Multiple Simulations</h2>
                            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                                Select simulations to export to CSV/Excel
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                    </button>
                </div>

                {/* Selection Info */}
                <div className="px-6 py-4 bg-blue-50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-800">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <span className="text-sm font-medium text-blue-900 dark:text-blue-300">
                                Selected: <strong>{selectedFiles.size}</strong> simulation(s)
                            </span>
                            {loadingData && (
                                <span className="text-xs text-blue-700 dark:text-blue-400 flex items-center gap-1">
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                    Loading data...
                                </span>
                            )}
                        </div>
                        {selectedFiles.size > 0 && (
                            <button
                                onClick={() => setSelectedFiles(new Set())}
                                className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium"
                            >
                                Clear All
                            </button>
                        )}
                    </div>
                </div>

                {/* Projects and Files List */}
                <div className="flex-1 overflow-y-auto p-6 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-gray-100 dark:[&::-webkit-scrollbar-track]:bg-gray-800 [&::-webkit-scrollbar-thumb]:bg-gray-300 dark:[&::-webkit-scrollbar-thumb]:bg-gray-600 [&::-webkit-scrollbar-thumb]:rounded-full">
                    {projects.length === 0 ? (
                        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                            <FileSpreadsheet className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600" />
                            <p>No projects with simulations found</p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {projects.map(project => {
                                const isExpanded = expandedProjects.has(project.id);
                                const filesInProject = project.files || [];
                                const selectedInProject = filesInProject.filter(f => selectedFiles.has(f.id)).length;

                                return (
                                    <div key={project.id} className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                                        {/* Project Header */}
                                        <div className="bg-gray-50 dark:bg-gray-700 p-4 flex items-center justify-between">
                                            <div className="flex items-center gap-3 flex-1">
                                                <button
                                                    onClick={() => toggleProjectExpansion(project.id)}
                                                    className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
                                                >
                                                    {isExpanded ? (
                                                        <CheckSquare className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                                                    ) : (
                                                        <Square className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                                                    )}
                                                </button>
                                                <div className="flex-1">
                                                    <h3 className="font-semibold text-gray-900 dark:text-gray-100">{project.name}</h3>
                                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                                        {filesInProject.length} file(s)
                                                        {selectedInProject > 0 && ` • ${selectedInProject} selected`}
                                                    </p>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => selectAllInProject(project)}
                                                className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium px-3 py-1 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded transition-colors"
                                            >
                                                {selectedInProject === filesInProject.length ? 'Deselect All' : 'Select All'}
                                            </button>
                                        </div>

                                        {/* Files List */}
                                        {isExpanded && (
                                            <div className="p-4 space-y-2 bg-white dark:bg-gray-800">
                                                {filesInProject.length === 0 ? (
                                                    <p className="text-sm text-gray-400 dark:text-gray-500 text-center py-4">
                                                        No files in this project
                                                    </p>
                                                ) : (
                                                    filesInProject.map(file => (
                                                        <label
                                                            key={file.id}
                                                            className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${selectedFiles.has(file.id)
                                                                ? 'border-blue-300 dark:border-blue-600 bg-blue-50 dark:bg-blue-900/20'
                                                                : 'border-gray-200 dark:border-gray-700 hover:border-blue-200 dark:hover:border-blue-700 hover:bg-gray-50 dark:hover:bg-gray-700'
                                                                }`}
                                                        >
                                                            <input
                                                                type="checkbox"
                                                                checked={selectedFiles.has(file.id)}
                                                                onChange={() => toggleFileSelection(file.id)}
                                                                className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                                                            />
                                                            <div className="flex-1 min-w-0">
                                                                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                                                                    {file.name}
                                                                </p>
                                                                {simulationData[file.id] && (
                                                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                                                        Simulation data loaded ✓
                                                                    </p>
                                                                )}
                                                            </div>
                                                        </label>
                                                    ))
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="border-t border-gray-200 dark:border-gray-700 p-6 bg-gray-50 dark:bg-gray-700">
                    <div className="flex items-center justify-between">
                        <div className="text-sm text-gray-600 dark:text-gray-300">
                            {selectedFiles.size > 0 ? (
                                <span>
                                    Ready to export <strong>{selectedFiles.size}</strong> simulation(s)
                                </span>
                            ) : (
                                <span>Select simulations to export</span>
                            )}
                        </div>
                        <div className="flex gap-3">
                            <button
                                onClick={onClose}
                                className="px-4 py-2 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={exportToCSV}
                                disabled={selectedFiles.size === 0 || loading || loadingData}
                                className="flex items-center gap-2 px-4 py-2 bg-green-600 dark:bg-green-700 text-white rounded-lg hover:bg-green-700 dark:hover:bg-green-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Exporting...
                                    </>
                                ) : (
                                    <>
                                        <Download className="w-4 h-4" />
                                        Export to CSV
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}