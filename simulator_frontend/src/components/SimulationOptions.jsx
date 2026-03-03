import React, { useState } from 'react';
import { Settings, X } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export default function SimulationOptions({ onClose, onSave, initialConfig }) {
    const { isDarkMode } = useTheme();
    const [config, setConfig] = useState(initialConfig || {
        N: 10,
        M: 2,
        NN_NAME: 'SimpleNN',
        R: 5,
        ROUNDS: 10,
        strategy: 'first',
        poison_operation: 'backdoor_blended',
        poison_intensity: 0.1,
        poison_percentage: 0.2,
        data_poison_protection: 'fedavg',
        target_class: '',
        no_flip: false,
        trigger_type: 'square',
        pattern_type: 'random',
        modification: 'green_tint',
        transform: 'rotation',
        watermark_type: 'apple'
    });

    const handleChange = (field, value) => {
        setConfig(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onSave(config);
    };

    // Determine which sub-parameters to show based on selected operation
    const op = config.poison_operation;
    const isBackdoor = op.startsWith('backdoor_');
    const showTargetClass = op === 'label_flip' || isBackdoor;
    const showNoFlip = isBackdoor;
    const showTriggerType = op === 'backdoor_badnets';
    const showPatternType = op === 'backdoor_blended';
    const showModification = op === 'semantic_backdoor';
    const showTransform = op === 'backdoor_edge_case';
    const showWatermarkType = op === 'backdoor_trojan';
    const hasSubParams = showTargetClass || showNoFlip || showTriggerType || showPatternType || showModification || showTransform || showWatermarkType;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto sidebar-scroll">
                <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Settings className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Simulation Options</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-6">
                    {/* FL Simulation Options */}
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-300 mb-4">Federated Learning Configuration</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Number of Clients (N)
                                </label>
                                <input
                                    type="number"
                                    value={config.N}
                                    onChange={(e) => handleChange('N', parseInt(e.target.value))}
                                    min="1"
                                    max="100"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Total number of participating clients</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Malicious Clients (M)
                                </label>
                                <input
                                    type="number"
                                    value={config.M}
                                    onChange={(e) => handleChange('M', parseInt(e.target.value))}
                                    min="0"
                                    max={config.N}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Number of malicious clients</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Training Rounds (ROUNDS)
                                </label>
                                <input
                                    type="number"
                                    value={config.ROUNDS}
                                    onChange={(e) => handleChange('ROUNDS', parseInt(e.target.value))}
                                    min="1"
                                    max="1000"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Total FL training rounds</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Neural Network Name
                                </label>
                                <input
                                    type="text"
                                    value={config.NN_NAME}
                                    onChange={(e) => handleChange('NN_NAME', e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    placeholder="e.g., SimpleNN, ResNet50"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Model architecture name</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Distribution Strategy
                                </label>
                                <select
                                    value={config.strategy}
                                    onChange={(e) => handleChange('strategy', e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                >
                                    <option value="first">First - Malicious at start</option>
                                    <option value="last">Last - Malicious at end</option>
                                    <option value="alternate">Alternate - Interleaved</option>
                                </select>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Client selection strategy</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Poisoned Data Rounds (R)
                                </label>
                                <input
                                    type="number"
                                    value={config.R}
                                    onChange={(e) => handleChange('R', parseInt(e.target.value))}
                                    min="0"
                                    max={config.ROUNDS}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Rounds using poisoned data</p>
                            </div>
                        </div>
                    </div>

                    {/* Data Poisoning Parameters */}
                    <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
                        <h3 className="text-lg font-semibold text-red-900 dark:text-red-300 mb-4">Data Poisoning Attack Parameters</h3>
                        <div className="grid grid-cols-3 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Poisoning Operation
                                </label>
                                <select
                                    value={config.poison_operation}
                                    onChange={(e) => handleChange('poison_operation', e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                >
                                    <option value="label_flip">🔄 Label Flip (dirty-label)</option>
                                    <option value="backdoor_badnets">🎯 BadNets Backdoor</option>
                                    <option value="backdoor_blended">🌀 Blended Backdoor</option>
                                    <option value="backdoor_sig">📡 SIG Backdoor (sinusoidal)</option>
                                    <option value="backdoor_trojan">🏴 Trojan Backdoor (watermark)</option>
                                    <option value="semantic_backdoor">🎨 Semantic Backdoor</option>
                                    <option value="backdoor_edge_case">🔀 Edge-case Backdoor</option>
                                </select>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Type of poisoning attack</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Intensity
                                </label>
                                <input
                                    type="number"
                                    value={config.poison_intensity}
                                    onChange={(e) => handleChange('poison_intensity', parseFloat(e.target.value))}
                                    min="0.01"
                                    max="1.0"
                                    step="0.01"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Attack intensity (0.01-1.0)</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Percentage
                                </label>
                                <input
                                    type="number"
                                    value={config.poison_percentage}
                                    onChange={(e) => handleChange('poison_percentage', parseFloat(e.target.value))}
                                    min="0.01"
                                    max="1.0"
                                    step="0.01"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                    required
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">% of data to poison (0.01-1.0)</p>
                            </div>
                        </div>

                        {/* Conditional sub-parameters based on selected operation */}
                        {hasSubParams && (
                            <div className="mt-4 p-3 bg-white dark:bg-gray-700 rounded-lg border border-red-200 dark:border-red-700">
                                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                                    ⚙️ Attack-Specific Parameters
                                </h4>
                                <div className="grid grid-cols-2 gap-4">
                                    {/* Target Class — for label_flip and all backdoor attacks */}
                                    {showTargetClass && (
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                Target Class
                                            </label>
                                            <input
                                                type="text"
                                                value={config.target_class || ''}
                                                onChange={(e) => handleChange('target_class', e.target.value)}
                                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                                placeholder="e.g., cat, 0 (leave empty for random)"
                                            />
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Target class for label flip (empty = random)</p>
                                        </div>
                                    )}

                                    {/* No Flip — for all backdoor attacks */}
                                    {showNoFlip && (
                                        <div className="flex items-center gap-3 self-end pb-2">
                                            <label className="flex items-center gap-2 cursor-pointer">
                                                <input
                                                    type="checkbox"
                                                    checked={config.no_flip || false}
                                                    onChange={(e) => handleChange('no_flip', e.target.checked)}
                                                    className="w-4 h-4 text-red-600 bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 rounded focus:ring-red-500"
                                                />
                                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                                    Don't flip labels
                                                </span>
                                            </label>
                                            <p className="text-xs text-gray-500 dark:text-gray-400">Keep original labels for backdoor images</p>
                                        </div>
                                    )}

                                    {/* Trigger Type — for backdoor_badnets */}
                                    {showTriggerType && (
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                Trigger Type
                                            </label>
                                            <select
                                                value={config.trigger_type || 'square'}
                                                onChange={(e) => handleChange('trigger_type', e.target.value)}
                                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                            >
                                                <option value="square">■ Square</option>
                                                <option value="cross">✚ Cross</option>
                                                <option value="L">⌐ L-shape</option>
                                                <option value="checkerboard">▦ Checkerboard</option>
                                            </select>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Shape of the BadNets trigger pattern</p>
                                        </div>
                                    )}

                                    {/* Pattern Type — for backdoor_blended */}
                                    {showPatternType && (
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                Pattern Type
                                            </label>
                                            <select
                                                value={config.pattern_type || 'random'}
                                                onChange={(e) => handleChange('pattern_type', e.target.value)}
                                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                            >
                                                <option value="random">🎲 Random noise</option>
                                                <option value="horizontal">═ Horizontal stripes</option>
                                                <option value="vertical">║ Vertical stripes</option>
                                                <option value="grid">▦ Grid pattern</option>
                                            </select>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Blended pattern mixed with images</p>
                                        </div>
                                    )}

                                    {/* Modification — for semantic_backdoor */}
                                    {showModification && (
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                Semantic Modification
                                            </label>
                                            <select
                                                value={config.modification || 'green_tint'}
                                                onChange={(e) => handleChange('modification', e.target.value)}
                                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                            >
                                                <option value="green_tint">🟢 Green tint</option>
                                                <option value="blue_tint">🔵 Blue tint</option>
                                                <option value="sepia">🟤 Sepia effect</option>
                                                <option value="high_contrast">⬛ High contrast</option>
                                                <option value="warm">🟠 Warm tones</option>
                                            </select>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Natural-looking semantic modification</p>
                                        </div>
                                    )}

                                    {/* Transform — for backdoor_edge_case */}
                                    {showTransform && (
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                Edge-case Transform
                                            </label>
                                            <select
                                                value={config.transform || 'rotation'}
                                                onChange={(e) => handleChange('transform', e.target.value)}
                                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                            >
                                                <option value="rotation">🔄 Rotation (15-45°)</option>
                                                <option value="flip_both">↕️ Flip both axes</option>
                                                <option value="negative">🔳 Partial negative</option>
                                                <option value="posterize">🎨 Posterize</option>
                                                <option value="solarize">☀️ Solarize</option>
                                            </select>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Rare transformation used as trigger</p>
                                        </div>
                                    )}

                                    {/* Watermark Type — for backdoor_trojan */}
                                    {showWatermarkType && (
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                Watermark Type
                                            </label>
                                            <select
                                                value={config.watermark_type || 'apple'}
                                                onChange={(e) => handleChange('watermark_type', e.target.value)}
                                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                            >
                                                <option value="apple">🍎 Apple</option>
                                                <option value="star">⭐ Star</option>
                                                <option value="circle">⚪ Circle</option>
                                                <option value="triangle">🔺 Triangle</option>
                                            </select>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Shape of the watermark used as trojan trigger</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        <div className="mt-3 p-3 bg-red-100 dark:bg-red-900/30 rounded text-sm text-red-800 dark:text-red-300">
                            <strong>Note:</strong> Data poisoning is automatically applied to simulate attacks.
                            Configure the attack parameters above.
                        </div>
                    </div>

                    {/* Data Poison Protection */}
                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                        <h3 className="text-lg font-semibold text-green-900 dark:text-green-300 mb-4">Data Poison Protection</h3>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                Aggregation Method
                            </label>
                            <select
                                value={config.data_poison_protection}
                                onChange={(e) => handleChange('data_poison_protection', e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                            >
                                <option value="fedavg">FedAvg - Standard (vulnerable to poisoning)</option>
                                <option value="krum">Krum - Selects closest update (99% attack elimination)</option>
                                <option value="trimmed_mean">Trimmed Mean - Removes extremes (resistant to label-flipping)</option>
                                <option value="median">Median - Resistant to 20% malicious clients</option>
                                <option value="foolsgold">FoolsGold - Sybil/Label Flip defense</option>
                                <option value="norm_clipping">Norm Clipping - Clips update norms (backdoor defense)</option>
                                <option value="trimmed_mean_krum">Trimmed Mean + Krum - Hybrid approach</option>
                                <option value="random">Random - Randomizes between Krum and Trimmed Mean</option>
                            </select>
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                Select the robust aggregation method to protect against data poisoning attacks
                            </p>
                        </div>
                        <div className="mt-3 p-3 bg-green-100 dark:bg-green-900/30 rounded text-sm text-green-800 dark:text-green-300">
                            <strong>Info:</strong> This parameter determines how metrics from multiple clients are aggregated.
                            Robust methods like Krum and Trimmed Mean provide protection against malicious clients.
                        </div>
                    </div>

                    {/* Action buttons */}
                    <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                        <button
                            type="button"
                            onClick={onClose}
                            className="flex-1 px-4 py-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="flex-1 px-4 py-3 bg-blue-600 dark:bg-blue-700 text-white rounded-lg font-semibold hover:bg-blue-700 dark:hover:bg-blue-600 transition-colors"
                        >
                            Save & Apply Configuration
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}