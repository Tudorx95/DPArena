import React, { useState, useEffect } from 'react';
import { ArrowLeft, GitCompare, Loader2 } from 'lucide-react';
import TopBar from '../components/TopBar';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function ComparePage({ onBack, token, activeProjectId, projects }) {
    const [simulations, setSimulations] = useState([]);
    const [sim1, setSim1] = useState(null);
    const [sim2, setSim2] = useState(null);
    const [results, setResults] = useState(null);
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

    const handleCompare = async () => {
        if (!sim1 || !sim2) {
            alert('Please select two simulations to compare');
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const res = await fetch(
                `${API_URL}/api/compare-simulations?sim1_id=${sim1}&sim2_id=${sim2}`,
                {
                    method: 'POST',
                    headers: authHeaders
                }
            );

            if (!res.ok) {
                throw new Error('Failed to compare simulations');
            }

            const data = await res.json();
            setResults(data);
        } catch (error) {
            console.error('Failed to compare simulations:', error);
            setError('Failed to compare simulations. Please try again.');
        } finally {
            setLoading(false);
        }
    };

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
                        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Compare Simulations</h1>
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
                {loading && !results && (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="w-8 h-8 animate-spin text-blue-600 dark:text-blue-400" />
                        <span className="ml-3 text-gray-600 dark:text-gray-300">Loading simulations...</span>
                    </div>
                )}

                {/* Simulation Selectors */}
                {!loading && simulations.length > 0 && (
                    <>
                        <div className="grid grid-cols-2 gap-4 mb-4">
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow border border-gray-200 dark:border-gray-700">
                                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                                    Simulation 1:
                                </label>
                                <select
                                    value={sim1 || ''}
                                    onChange={(e) => setSim1(Number(e.target.value))}
                                    className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                >
                                    <option value="">-- Choose Simulation --</option>
                                    {simulations.map(s => (
                                        <option key={s.id} value={s.id}>
                                            {new Date(s.completed_at).toLocaleString()} - {getFileName(s.file_id)}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow border border-gray-200 dark:border-gray-700">
                                <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                                    Simulation 2:
                                </label>
                                <select
                                    value={sim2 || ''}
                                    onChange={(e) => setSim2(Number(e.target.value))}
                                    className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                >
                                    <option value="">-- Choose Simulation --</option>
                                    {simulations.map(s => (
                                        <option key={s.id} value={s.id}>
                                            {new Date(s.completed_at).toLocaleString()} - {getFileName(s.file_id)}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Compare Button */}
                        <button
                            onClick={handleCompare}
                            disabled={!sim1 || !sim2 || loading}
                            className="w-full bg-blue-600 dark:bg-blue-700 text-white py-3 rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors shadow"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Comparing...
                                </>
                            ) : (
                                <>
                                    <GitCompare className="w-5 h-5" />
                                    Compare Simulations
                                </>
                            )}
                        </button>
                    </>
                )}

                {/* No Simulations Message */}
                {!loading && simulations.length === 0 && (
                    <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-300 px-4 py-3 rounded">
                        <p className="font-medium">No completed simulations found</p>
                        <p className="text-sm mt-1">Run some simulations first to be able to compare them.</p>
                    </div>
                )}

                {/* Results */}
                {results && (
                    <div className="grid grid-cols-2 gap-4 mt-6">
                        <ResultBox
                            title={`Simulation 1 (${getFileName(results.simulation1.file_id)})`}
                            data={results.simulation1}
                        />
                        <ResultBox
                            title={`Simulation 2 (${getFileName(results.simulation2.file_id)})`}
                            data={results.simulation2}
                            reversed
                        />
                    </div>
                )}
            </div>
        </div>
    );
}

function ResultBox({ title, data, reversed = false }) {
    const flexDir = reversed ? 'flex flex-row-reverse' : 'flex';
    const analysis = data.results?.analysis || {};
    const summary = data.results?.summary || 'No summary available';
    const config = data.config || {};

    const getPoisonOperationLabel = (op) => {
        const map = {
            'label_flip': '🔄 Label Flip',
            'backdoor_badnets': '🎯 BadNets Backdoor',
            'backdoor_blended': '🌀 Blended Backdoor',
            'backdoor_sig': '📡 SIG Backdoor',
            'backdoor_trojan': '🏴 Trojan Backdoor',
            'semantic_backdoor': '🎨 Semantic Backdoor',
            'backdoor_edge_case': '🔀 Edge-case Backdoor'
        };
        if (!op) return 'N/A';
        if (map[op]) return map[op];
        if (op.startsWith('@')) return `🧪 ${op} (custom)`;
        return op;
    };

    const getAggregationLabel = (agg) => {
        const map = {
            'fedavg': '⚖️ FedAvg',
            'krum': '🛡️ Krum',
            'trimmed_mean': '✂️ Trimmed Mean',
            'median': '📊 Median',
            'foolsgold': '🥇 FoolsGold',
            'norm_clipping': '📏 Norm Clipping',
            'trimmed_mean_krum': '🔗 Trimmed Mean + Krum',
            'random': '🎲 Random'
        };
        if (!agg) return 'N/A';
        if (map[agg]) return map[agg];
        if (agg.startsWith('@')) return `🧩 ${agg} (custom)`;
        return agg;
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow border border-gray-200 dark:border-gray-700">
            <h3 className={`text-lg font-bold mb-4 text-gray-800 dark:text-gray-100 border-b dark:border-gray-700 pb-2 ${reversed ? 'text-right' : ''}`}>{title}</h3>

            {/* Configuration */}
            <div className="mb-4">
                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">📊 FL Configuration:</h4>
                <div className="space-y-1 text-sm text-gray-600 dark:text-gray-300 bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
                    <div className={`${flexDir} justify-between`}>
                        <strong>Total Clients (N)</strong>
                        <span className="font-mono text-blue-700 dark:text-blue-400">{config.N || 'N/A'}</span>
                    </div>
                    <div className={`${flexDir} justify-between`}>
                        <strong>Malicious Clients (M)</strong>
                        <span className="font-mono text-red-600 dark:text-red-400">{config.M || 'N/A'}</span>
                    </div>
                    <div className={`${flexDir} justify-between`}>
                        <strong>Neural Network</strong>
                        <span className="font-mono">{config.NN_NAME || 'N/A'}</span>
                    </div>
                    <div className={`${flexDir} justify-between`}>
                        <strong>Training Rounds</strong>
                        <span className="font-mono text-green-700 dark:text-green-400">{config.ROUNDS || 'N/A'}</span>
                    </div>
                    <div className={`${flexDir} justify-between`}>
                        <strong>Poisoned Rounds (R)</strong>
                        <span className="font-mono text-orange-600 dark:text-orange-400">{config.R || 'N/A'}</span>
                    </div>
                    <div className={`${flexDir} justify-between`}>
                        <strong>Strategy</strong>
                        <span className="font-mono">{config.strategy || 'N/A'}</span>
                    </div>
                </div>
            </div>

            {/* Data Poisoning Configuration */}
            <div className="mb-4">
                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">🦠 Data Poisoning Attack:</h4>
                <div className="space-y-2 text-sm bg-red-50 dark:bg-red-900/20 p-3 rounded border border-red-200 dark:border-red-800">
                    {/* Poisoning Operation */}
                    <div className="p-2 bg-white dark:bg-gray-700 rounded border border-red-200 dark:border-red-700">
                        <div className={`${flexDir} justify-between items-center`}>
                            <strong className="text-gray-700 dark:text-gray-300">Operation</strong>
                            <span className="font-mono text-red-700 dark:text-red-400">
                                {getPoisonOperationLabel(config.poison_operation)}
                            </span>
                        </div>
                    </div>

                    {/* Attack Intensity */}
                    <div className="p-2 bg-white dark:bg-gray-700 rounded border border-orange-200 dark:border-orange-700">
                        <div className={`${flexDir} justify-between items-center mb-1`}>
                            <strong className="text-gray-700 dark:text-gray-300">Intensity</strong>
                            <span className="font-mono text-orange-700 dark:text-orange-400 text-base font-bold">
                                {config.poison_intensity ? (config.poison_intensity * 100).toFixed(1) : 'N/A'}%
                            </span>
                        </div>
                        {config.poison_intensity !== undefined && (
                            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2 mt-1">
                                <div
                                    className={`h-2 rounded-full transition-all ${reversed ? 'bg-gradient-to-r from-orange-400 to-red-500' : 'bg-gradient-to-r from-orange-400 to-red-500 ml-auto'}`}
                                    style={{ width: `${config.poison_intensity * 100}%` }}
                                />
                            </div>
                        )}
                    </div>

                    {/* Poisoned Data Percentage */}
                    <div className="p-2 bg-white dark:bg-gray-700 rounded border border-red-300 dark:border-red-700">
                        <div className={`${flexDir} justify-between items-center mb-1`}>
                            <strong className="text-gray-700 dark:text-gray-300">Data Percentage</strong>
                            <span className="font-mono text-red-700 dark:text-red-400 text-base font-bold">
                                {config.poison_percentage ? (config.poison_percentage * 100).toFixed(1) : 'N/A'}%
                            </span>
                        </div>
                        {config.poison_percentage !== undefined && (
                            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2 mt-1">
                                <div
                                    className={`h-2 rounded-full transition-all ${reversed ? 'bg-gradient-to-r from-red-500 to-red-700' : 'bg-gradient-to-r from-red-500 to-red-700 ml-auto'}`}
                                    style={{ width: `${config.poison_percentage * 100}%` }}
                                />
                            </div>
                        )}
                    </div>

                    {/* Attack Summary */}
                    {config.poison_operation && (
                        <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded border border-red-300 dark:border-red-700 text-xs text-red-800 dark:text-red-300">
                            <strong>Summary:</strong> Using <strong>{config.poison_operation}</strong> with{' '}
                            <strong>{config.poison_intensity ? (config.poison_intensity * 100).toFixed(1) : 0}%</strong> intensity on{' '}
                            <strong>{config.poison_percentage ? (config.poison_percentage * 100).toFixed(1) : 0}%</strong> of data{' '}
                            for <strong>{config.R || 0}</strong> rounds with <strong>{config.M || 0}</strong> malicious clients.
                        </div>
                    )}

                    {/* Aggregation Function */}
                    <div className="p-2 bg-white dark:bg-gray-700 rounded border border-green-300 dark:border-green-700">
                        <div className={`${flexDir} justify-between items-center`}>
                            <strong className="text-gray-700 dark:text-gray-300">Aggregation Function</strong>
                            <span className="font-mono text-green-700 dark:text-green-400 font-bold">
                                {getAggregationLabel(config.data_poison_protection)}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Results */}
            <div className="mb-4">
                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">📈 Simulation Results:</h4>
                <div className="space-y-2 text-sm">
                    <div className={`${flexDir} justify-between p-2 bg-purple-50 dark:bg-purple-900/20 rounded border border-purple-200 dark:border-purple-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Init Accuracy</strong>
                        <span className="font-mono text-purple-700 dark:text-purple-400 text-base font-bold">
                            {analysis.init_accuracy?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Clean Accuracy</strong>
                        <span className="font-mono text-green-700 dark:text-green-400 text-base font-bold">
                            {analysis.clean_accuracy?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-emerald-50 dark:bg-emerald-900/20 rounded border border-emerald-200 dark:border-emerald-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Clean + DP Protection Accuracy</strong>
                        <span className="font-mono text-emerald-700 dark:text-emerald-400 text-base font-bold">
                            {analysis.clean_dp_accuracy?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Poisoned Accuracy</strong>
                        <span className="font-mono text-red-700 dark:text-red-400 text-base font-bold">
                            {analysis.poisoned_accuracy?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-teal-50 dark:bg-teal-900/20 rounded border border-teal-200 dark:border-teal-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Poisoned + DP Protection Accuracy</strong>
                        <span className="font-mono text-teal-700 dark:text-teal-400 text-base font-bold">
                            {analysis.poisoned_dp_accuracy?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-orange-50 dark:bg-orange-900/20 rounded border border-orange-200 dark:border-orange-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Drop (Clean - Poisoned)</strong>
                        <span className="font-mono text-orange-700 dark:text-orange-400 text-base font-bold">
                            {analysis.accuracy_drop?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Drop (Clean - Init)</strong>
                        <span className="font-mono text-blue-700 dark:text-blue-400 text-base font-bold">
                            {analysis.drop_clean_init?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Drop (Clean DP - Init)</strong>
                        <span className="font-mono text-blue-700 dark:text-blue-400 text-base font-bold">
                            {analysis.drop_clean_dp_init?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Drop (Poisoned - Init)</strong>
                        <span className="font-mono text-blue-700 dark:text-blue-400 text-base font-bold">
                            {analysis.drop_poison_init?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">Drop (Poisoned DP - Init)</strong>
                        <span className="font-mono text-blue-700 dark:text-blue-400 text-base font-bold">
                            {analysis.drop_poison_dp_init?.toFixed(4) || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-cyan-50 dark:bg-cyan-900/20 rounded border border-cyan-200 dark:border-cyan-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">DP Protection Method</strong>
                        <span className="font-mono text-cyan-700 dark:text-cyan-400 text-base font-bold">
                            {analysis.data_poison_protection_method || 'N/A'}
                        </span>
                    </div>
                    <div className={`${flexDir} justify-between p-2 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800`}>
                        <strong className="text-gray-700 dark:text-gray-300">GPU Used</strong>
                        <span className="font-mono text-blue-700 dark:text-blue-400">
                            {analysis.gpu_used || 'N/A'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Summary */}
            <div>
                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">📝 Summary:</h4>
                <pre className="text-xs bg-gray-900 dark:bg-gray-950 text-gray-100 dark:text-gray-200 p-3 rounded overflow-auto max-h-64 font-mono border border-gray-700 dark:border-gray-600">
                    {summary}
                </pre>
            </div>

            {/* Timestamp */}
            {data.completed_at && (
                <div className="mt-4 pt-3 border-t dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
                    ⏱️ Completed: {new Date(data.completed_at).toLocaleString()}
                </div>
            )}
        </div>
    );
}