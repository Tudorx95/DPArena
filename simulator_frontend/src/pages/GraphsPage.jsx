import React, { useState, useEffect } from 'react';
import { ArrowLeft, BarChart3, Loader2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import TopBar from '../components/TopBar';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function GraphsPage({ onBack, token, activeProjectId, projects }) {
    const [simulations, setSimulations] = useState([]);
    const [selectedSimulations, setSelectedSimulations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const activeProject = projects?.find(p => String(p.id) === String(activeProjectId));

    const getFileName = (fileId) => {
        if (!projects) return 'Unknown File';
        for (const project of projects) {
            const file = project.files?.find(f => String(f.id) === String(fileId));
            if (file) return file.name;
        }
        return 'Unknown File';
    };

    // Auth headers
    const authHeaders = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };

    // Load simulations when component mounts
    useEffect(() => {
        if (activeProjectId) {
            loadSimulations();
        }
    }, [activeProjectId]);

    const loadSimulations = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(
                `${API_URL}/api/projects/${activeProjectId}/simulations`,
                { headers: authHeaders }
            );

            if (!res.ok) {
                throw new Error('Failed to load simulations');
            }

            const data = await res.json();
            setSimulations(data.simulations || []);
        } catch (error) {
            console.error('Failed to load simulations:', error);
            setError('Failed to load simulations. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const toggleSimulation = (simId) => {
        setSelectedSimulations(prev => {
            if (prev.includes(simId)) {
                return prev.filter(id => id !== simId);
            } else {
                return [...prev, simId];
            }
        });
    };

    // Prepare chart data from selected simulations - grouped by metric
    const prepareChartData = () => {
        if (selectedSimulations.length === 0) return [];

        const selectedSims = simulations.filter(sim => selectedSimulations.includes(sim.id));

        // Create separate data points for each metric type
        const metrics = [
            { key: 'init_accuracy', label: 'Init Accuracy', color: '#8B5CF6' },
            { key: 'clean_accuracy', label: 'Clean Accuracy', color: '#10B981' },
            { key: 'clean_dp_accuracy', label: 'Clean + DP Protection', color: '#059669' },
            { key: 'poisoned_accuracy', label: 'Poisoned Accuracy', color: '#EF4444' },
            { key: 'poisoned_dp_accuracy', label: 'Poisoned + DP Protection', color: '#06B6D4' }
        ];

        const chartData = metrics.map(metric => {
            const dataPoint = { metric: metric.label, color: metric.color };

            selectedSims.forEach((sim, index) => {
                const analysis = sim.results?.analysis || {};
                const value = analysis[metric.key] || 0;
                dataPoint[`sim${index + 1}`] = value;
                dataPoint[`sim${index + 1}_name`] = `Sim ${index + 1}`;
                dataPoint[`sim${index + 1}_taskId`] = getFileName(sim.file_id);
            });

            return dataPoint;
        });

        return chartData;
    };

    const chartData = prepareChartData();
    const selectedSims = simulations.filter(sim => selectedSimulations.includes(sim.id));

    // Color palette for bars - one color per simulation
    const simColors = ['#3B82F6', '#F59E0B', '#EC4899', '#8B5CF6', '#14B8A6', '#F97316', '#06B6D4', '#84CC16'];

    return (
        <div className="h-screen bg-gray-50 dark:bg-gray-900 flex flex-col overflow-hidden">
            {/* TopBar with theme toggle */}
            <TopBar disableNavigation={true} />

            <div className="flex-1 p-6 overflow-auto page-scroll">
                {/* Header */}
                <div className="relative flex items-center mb-6 min-h-[40px]">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={onBack}
                            className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
                        >
                            <ArrowLeft className="w-5 h-5 text-gray-900 dark:text-gray-100" />
                        </button>
                        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Simulation Graphs</h1>
                    </div>
                    {activeProject && (
                        <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1.5">
                            <span className="text-lg font-semibold text-blue-600 dark:text-blue-400">📁 {activeProject.name}</span>
                        </div>
                    )}
                </div>

                {/* Error Message */}
                {error && (
                    <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded mb-4">
                        {error}
                    </div>
                )}

                {/* Loading State */}
                {loading && (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="w-8 h-8 animate-spin text-blue-600 dark:text-blue-400" />
                        <span className="ml-3 text-gray-600 dark:text-gray-300">Loading simulations...</span>
                    </div>
                )}

                {/* Simulation Selection */}
                {!loading && simulations.length > 0 && (
                    <div className="mb-6">
                        <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-3">
                            Select Simulations to Display
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                            {simulations.map(sim => (
                                <div
                                    key={sim.id}
                                    onClick={() => toggleSimulation(sim.id)}
                                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${selectedSimulations.includes(sim.id)
                                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30'
                                        : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-gray-300 dark:hover:border-gray-600'
                                        }`}
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                                File: {getFileName(sim.file_id)}
                                            </div>
                                            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                                {new Date(sim.completed_at).toLocaleString()}
                                            </div>
                                            <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                                                <div>N: {sim.config?.N} | M: {sim.config?.M}</div>
                                                <div>Rounds: {sim.config?.ROUNDS}</div>
                                            </div>
                                        </div>
                                        <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${selectedSimulations.includes(sim.id)
                                            ? 'border-blue-500 bg-blue-500'
                                            : 'border-gray-300 dark:border-gray-600'
                                            }`}>
                                            {selectedSimulations.includes(sim.id) && (
                                                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7" />
                                                </svg>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* No Simulations Message */}
                {!loading && simulations.length === 0 && (
                    <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-300 px-4 py-3 rounded">
                        <p className="font-medium">No completed simulations found</p>
                        <p className="text-sm mt-1">Run some simulations first to view their graphs.</p>
                    </div>
                )}

                {/* Charts */}
                {selectedSimulations.length > 0 && (
                    <div className="space-y-6">
                        {/* Accuracy Comparison Chart */}
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow border border-gray-200 dark:border-gray-700">
                            <h3 className="text-lg font-bold mb-4 text-gray-800 dark:text-gray-100 flex items-center gap-2">
                                <BarChart3 className="w-5 h-5" />
                                Accuracy Comparison - Grouped by Metric
                            </h3>
                            {chartData.length > 0 && selectedSims.length > 0 ? (
                                <ResponsiveContainer width="100%" height={500}>
                                    <BarChart data={chartData} barGap={5} barCategoryGap="20%">
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis
                                            dataKey="metric"
                                            stroke="#9CA3AF"
                                            tick={{ fill: '#9CA3AF', fontSize: 12 }}
                                        />
                                        <YAxis
                                            stroke="#9CA3AF"
                                            tick={{ fill: '#9CA3AF' }}
                                            domain={[0, 1]}
                                            label={{
                                                value: 'Accuracy',
                                                angle: -90,
                                                position: 'insideLeft',
                                                style: { fill: '#9CA3AF' }
                                            }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1F2937',
                                                border: '1px solid #374151',
                                                borderRadius: '8px',
                                                color: '#F9FAFB'
                                            }}
                                            formatter={(value, name) => {
                                                return [value?.toFixed(4) || '0.0000', name];
                                            }}
                                        />
                                        <Legend
                                            wrapperStyle={{ paddingTop: '20px' }}
                                        />
                                        {selectedSims.map((sim, index) => (
                                            <Bar
                                                key={`sim${index + 1}`}
                                                dataKey={`sim${index + 1}`}
                                                fill={simColors[index % simColors.length]}
                                                name={`Simulation ${index + 1} (${getFileName(sim.file_id)})`}
                                                radius={[4, 4, 0, 0]}
                                                minPointSize={5}
                                            />
                                        ))}
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex items-center justify-center h-[500px] text-gray-500 dark:text-gray-400">
                                    <p>No data available for selected simulations</p>
                                </div>
                            )}
                        </div>

                        {/* Detailed Comparison Table */}
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow border border-gray-200 dark:border-gray-700">
                            <h3 className="text-lg font-bold mb-4 text-gray-800 dark:text-gray-100">Detailed Metrics</h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b dark:border-gray-700">
                                            <th className="text-left py-3 px-4 text-gray-700 dark:text-gray-300">Simulation</th>
                                            <th className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">Init Acc.</th>
                                            <th className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">Clean Acc.</th>
                                            <th className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">Clean+DP Acc.</th>
                                            <th className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">Poisoned Acc.</th>
                                            <th className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">Poisoned+DP Acc.</th>
                                            <th className="text-right py-3 px-4 text-gray-700 dark:text-gray-300">Accuracy Drop</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {simulations
                                            .filter(sim => selectedSimulations.includes(sim.id))
                                            .map((sim, index) => {
                                                const analysis = sim.results?.analysis || {};
                                                return (
                                                    <tr key={sim.id} className="border-b dark:border-gray-700">
                                                        <td className="py-3 px-4 text-gray-700 dark:text-gray-300">
                                                            Sim {index + 1}
                                                            <div className="text-xs text-gray-500 dark:text-gray-400">
                                                                {getFileName(sim.file_id)}
                                                            </div>
                                                        </td>
                                                        <td className="text-right py-3 px-4 font-mono text-purple-600 dark:text-purple-400">
                                                            {analysis.init_accuracy?.toFixed(4) || 'N/A'}
                                                        </td>
                                                        <td className="text-right py-3 px-4 font-mono text-green-600 dark:text-green-400">
                                                            {analysis.clean_accuracy?.toFixed(4) || 'N/A'}
                                                        </td>
                                                        <td className="text-right py-3 px-4 font-mono text-emerald-600 dark:text-emerald-400">
                                                            {analysis.clean_dp_accuracy?.toFixed(4) || 'N/A'}
                                                        </td>
                                                        <td className="text-right py-3 px-4 font-mono text-red-600 dark:text-red-400">
                                                            {analysis.poisoned_accuracy?.toFixed(4) || 'N/A'}
                                                        </td>
                                                        <td className="text-right py-3 px-4 font-mono text-cyan-600 dark:text-cyan-400">
                                                            {analysis.poisoned_dp_accuracy?.toFixed(4) || 'N/A'}
                                                        </td>
                                                        <td className="text-right py-3 px-4 font-mono text-orange-600 dark:text-orange-400">
                                                            {analysis.accuracy_drop?.toFixed(4) || 'N/A'}
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}


            </div>
        </div>
    );
}
