import React, { useState, useRef, useEffect } from 'react';
import { Settings, X, Plus, Trash2, ChevronDown, Loader2 } from 'lucide-react';
import CustomAggregationModal from './CustomAggregationModal';
import CustomPoisoningModal from './CustomPoisoningModal';
import { useTheme } from '../context/ThemeContext';

export default function SimulationOptions({ onClose, onSave, initialConfig, apiUrl, token }) {
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

    // Custom aggregation functions state (fetched from API on mount)
    const [customFunctions, setCustomFunctions] = useState([]);
    const [showCustomModal, setShowCustomModal] = useState(false);

    // Custom poisoning functions state (fetched from API on mount)
    const [customPoisoningFunctions, setCustomPoisoningFunctions] = useState([]);
    const [showCustomPoisoningModal, setShowCustomPoisoningModal] = useState(false);

    // Dropdown states
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const dropdownRef = useRef(null);

    const [isPoisonDropdownOpen, setIsPoisonDropdownOpen] = useState(false);
    const poisonDropdownRef = useRef(null);

    const [isDeletingParams, setIsDeletingParams] = useState({ funcType: null, funcName: null });

    useEffect(() => {
        function handleClickOutside(event) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsDropdownOpen(false);
            }
            if (poisonDropdownRef.current && !poisonDropdownRef.current.contains(event.target)) {
                setIsPoisonDropdownOpen(false);
            }
        }
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        const fetchCustomFunctions = async () => {
            if (!apiUrl || !token) return;
            try {
                const response = await fetch(`${apiUrl}/api/custom-functions`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                if (response.ok) {
                    const data = await response.json();
                    const agg = data.aggregation || [];
                    const pois = data.poisoning || [];

                    setCustomFunctions(agg);
                    setCustomPoisoningFunctions(pois);
                }
            } catch (err) {
                console.error("Failed to fetch custom functions", err);
            }
        };
        fetchCustomFunctions();
    }, [apiUrl, token]);

    const predefinedAggregationOptions = [
        { value: 'fedavg', label: 'FedAvg - Standard (vulnerable to poisoning)' },
        { value: 'krum', label: 'Krum - Selects closest update (99% attack elimination)' },
        { value: 'trimmed_mean', label: 'Trimmed Mean - Removes extremes (resistant to label-flipping)' },
        { value: 'median', label: 'Median - Resistant to 20% malicious clients' },
        { value: 'foolsgold', label: 'FoolsGold - Sybil/Label Flip defense' },
        { value: 'norm_clipping', label: 'Norm Clipping - Clips update norms (backdoor defense)' },
        { value: 'trimmed_mean_krum', label: 'Trimmed Mean + Krum - Hybrid approach' },
        { value: 'random', label: 'Random - Randomizes between Krum and Trimmed Mean' }
    ];

    const predefinedPoisoningOptions = [
        { value: 'label_flip', label: '🔄 Label Flip (dirty-label)' },
        { value: 'backdoor_badnets', label: '🎯 BadNets Backdoor' },
        { value: 'backdoor_blended', label: '🌀 Blended Backdoor' },
        { value: 'backdoor_sig', label: '📡 SIG Backdoor (sinusoidal)' },
        { value: 'backdoor_trojan', label: '🏴 Trojan Backdoor (watermark)' },
        { value: 'semantic_backdoor', label: '🎨 Semantic Backdoor' },
        { value: 'backdoor_edge_case', label: '🔀 Edge-case Backdoor' }
    ];

    const getCurrentAggregationLabel = () => {
        const val = config.data_poison_protection;
        const builtin = predefinedAggregationOptions.find(opt => opt.value === val);
        if (builtin) return builtin.label;
        if (val?.startsWith('@')) {
            return `🧩 ${val} (custom)`;
        }
        return val;
    };

    const getCurrentPoisoningLabel = () => {
        const val = config.poison_operation;
        const builtin = predefinedPoisoningOptions.find(opt => opt.value === val);
        if (builtin) return builtin.label;
        if (val?.startsWith('@')) {
            return `🧪 ${val} (custom)`;
        }
        return val;
    };

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

    // Custom aggregation handlers
    const handleAddCustomFunction = (funcData) => {
        const updated = [...customFunctions.filter(f => f.name !== funcData.name), funcData];
        setCustomFunctions(updated);
        handleChange('data_poison_protection', `@${funcData.name}`);
        setShowCustomModal(false);
    };

    const handleRemoveCustomFunction = async (funcName) => {
        setIsDeletingParams({ funcType: 'aggregation', funcName });
        try {
            const res = await fetch(`${apiUrl}/api/custom-function/aggregation/${funcName}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const updated = customFunctions.filter(f => f.name !== funcName);
                setCustomFunctions(updated);
                if (config.data_poison_protection === `@${funcName}`) {
                    handleChange('data_poison_protection', 'fedavg');
                }
            } else {
                console.error("Failed to delete aggregation function");
            }
        } catch (e) { console.error(e); }
        finally { setIsDeletingParams({ funcType: null, funcName: null }); }
    };

    const handleAddCustomPoisoningFunction = (funcData) => {
        const updated = [...customPoisoningFunctions.filter(f => f.name !== funcData.name), funcData];
        setCustomPoisoningFunctions(updated);
        handleChange('poison_operation', `@${funcData.name}`);
        setShowCustomPoisoningModal(false);
    };

    const handleRemoveCustomPoisoningFunction = async (funcName) => {
        setIsDeletingParams({ funcType: 'poisoning', funcName });
        try {
            const res = await fetch(`${apiUrl}/api/custom-function/poisoning/${funcName}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const updated = customPoisoningFunctions.filter(f => f.name !== funcName);
                setCustomPoisoningFunctions(updated);
                if (config.poison_operation === `@${funcName}`) {
                    handleChange('poison_operation', 'label_flip');
                }
            } else {
                console.error("Failed to delete poisoning function");
            }
        } catch (e) { console.error(e); }
        finally { setIsDeletingParams({ funcType: null, funcName: null }); }
    };

    // Determine which sub-parameters to show based on selected operation
    const op = config.poison_operation;
    const isBackdoor = op.startsWith('backdoor_');
    const showTargetClass = op === 'label_flip' || isBackdoor || op.startsWith('@');
    const showNoFlip = isBackdoor;
    const showTriggerType = op === 'backdoor_badnets';
    const showPatternType = op === 'backdoor_blended';
    const showModification = op === 'semantic_backdoor';
    const showTransform = op === 'backdoor_edge_case';
    const showWatermarkType = op === 'backdoor_trojan';
    const hasSubParams = showTargetClass || showNoFlip || showTriggerType || showPatternType || showModification || showTransform || showWatermarkType;

    // Attacks that use poison_intensity as a meaningful hyperparameter:
    // - backdoor_blended: blending ratio (alpha) of the pattern overlay
    // - backdoor_sig: amplitude of the sinusoidal signal
    // - backdoor_badnets: controls trigger size/strength
    // - backdoor_trojan: opacity of the watermark
    // Attacks that do NOT use intensity:
    // - label_flip: just flips the label, no numeric intensity
    // - semantic_backdoor: applies a fixed semantic transformation (no ratio)
    // - backdoor_edge_case: applies a fixed geometric transform (no ratio)
    // - custom (@...) functions: unknown, keep enabled for safety
    const INTENSITY_ATTACKS = new Set(['backdoor_blended', 'backdoor_sig', 'backdoor_badnets', 'backdoor_trojan']);
    const intensityEnabled = INTENSITY_ATTACKS.has(op) || op.startsWith('@');
    const intensityDisabledReason = !intensityEnabled
        ? op === 'label_flip'
            ? 'Label Flip does not use intensity — it only flips labels'
            : op === 'semantic_backdoor'
                ? 'Semantic Backdoor uses a fixed visual modification — intensity has no effect'
                : op === 'backdoor_edge_case'
                    ? 'Edge-case Backdoor uses a fixed geometric transform — intensity has no effect'
                    : 'This attack does not use intensity'
        : null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto sidebar-scroll">
                <div className="sticky top-0 z-50 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex items-center justify-between">
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
                        <div className="grid grid-cols-12 gap-4">
                            <div className="col-span-12 sm:col-span-6">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Poisoning Operation
                                </label>
                                <div className="flex items-center gap-2">
                                    <div className="relative flex-1 min-w-0" ref={poisonDropdownRef}>
                                        <button
                                            type="button"
                                            onClick={() => setIsPoisonDropdownOpen(!isPoisonDropdownOpen)}
                                            className="w-full text-left px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 flex justify-between items-center min-w-0"
                                        >
                                            <span className="truncate flex-1 min-w-0 pr-4">{getCurrentPoisoningLabel()}</span>
                                            <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isPoisonDropdownOpen ? 'rotate-180' : ''}`} />
                                        </button>

                                        {isPoisonDropdownOpen && (
                                            <div className="absolute z-40 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl max-h-56 overflow-y-auto sidebar-scroll">
                                                {predefinedPoisoningOptions.map(opt => (
                                                    <button
                                                        key={opt.value}
                                                        type="button"
                                                        onClick={() => {
                                                            handleChange('poison_operation', opt.value);
                                                            setIsPoisonDropdownOpen(false);
                                                        }}
                                                        className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${config.poison_operation === opt.value ? 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 font-medium' : 'text-gray-700 dark:text-gray-200'}`}
                                                    >
                                                        {opt.label}
                                                    </button>
                                                ))}

                                                {customPoisoningFunctions.length > 0 && (
                                                    <>
                                                        <div className="px-3 py-1.5 text-xs font-semibold text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-900 border-t border-b border-gray-200 dark:border-gray-700 uppercase tracking-wider">
                                                            ── Custom Operations ──
                                                        </div>
                                                        {customPoisoningFunctions.map(fn => (
                                                            <button
                                                                key={fn.name}
                                                                type="button"
                                                                onClick={() => {
                                                                    handleChange('poison_operation', `@${fn.name}`);
                                                                    setIsPoisonDropdownOpen(false);
                                                                }}
                                                                className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2 ${config.poison_operation === `@${fn.name}` ? 'bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 font-medium' : 'text-gray-700 dark:text-gray-200'}`}
                                                            >
                                                                <span>🧪</span> @{fn.name} (custom)
                                                            </button>
                                                        ))}
                                                    </>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    <button
                                        type="button"
                                        onClick={() => setShowCustomPoisoningModal(true)}
                                        className="flex-shrink-0 p-2 bg-orange-100 hover:bg-orange-200 dark:bg-orange-900/40 dark:hover:bg-orange-900/60 text-orange-700 dark:text-orange-300 rounded-lg transition-colors"
                                        title="Add custom data poisoning function"
                                    >
                                        <Plus className="w-5 h-5" />
                                    </button>
                                </div>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Type of poisoning attack</p>
                            </div>

                            <div className="col-span-6 sm:col-span-3">
                                <label className={`block text-sm font-medium mb-2 ${intensityEnabled
                                    ? 'text-gray-700 dark:text-gray-300'
                                    : 'text-gray-400 dark:text-gray-500'
                                    }`}>
                                    Intensity
                                    {!intensityEnabled && (
                                        <span className="ml-1.5 text-xs font-normal text-gray-400 dark:text-gray-500">— N/A</span>
                                    )}
                                </label>
                                <div title={intensityDisabledReason || ''}>
                                    <input
                                        type="number"
                                        value={config.poison_intensity}
                                        onChange={(e) => handleChange('poison_intensity', parseFloat(e.target.value))}
                                        min="0.01"
                                        max="1.0"
                                        step="0.01"
                                        disabled={!intensityEnabled}
                                        className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent transition-colors ${intensityEnabled
                                            ? 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
                                            : 'border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 cursor-not-allowed opacity-60'
                                            }`}
                                        required={intensityEnabled}
                                    />
                                </div>
                                <p className={`text-xs mt-1 ${intensityEnabled
                                    ? 'text-gray-500 dark:text-gray-400'
                                    : 'text-gray-400 dark:text-gray-500 italic'
                                    }`}>
                                    {intensityEnabled
                                        ? 'Attack intensity (0.01-1.0)'
                                        : 'Not used by this attack'}
                                </p>
                            </div>

                            <div className="col-span-6 sm:col-span-3">
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

                            {/* List of custom poisoning functions with remove buttons (Full Width) */}
                            {customPoisoningFunctions.length > 0 && (
                                <div className="col-span-12 mt-2">
                                    <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Custom operations:</p>
                                    <div className="space-y-1 max-h-36 overflow-y-auto sidebar-scroll pr-1">
                                        {customPoisoningFunctions.map(fn => (
                                            <div key={fn.name} className="flex items-center justify-between px-3 py-1.5 bg-white dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600">
                                                <span className="text-sm font-mono text-orange-700 dark:text-orange-300">
                                                    @{fn.name}
                                                </span>
                                                <button
                                                    type="button"
                                                    onClick={() => handleRemoveCustomPoisoningFunction(fn.name)}
                                                    disabled={isDeletingParams.funcName === fn.name && isDeletingParams.funcType === 'poisoning'}
                                                    className="p-1 text-red-400 hover:text-red-600 dark:text-red-500 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors disabled:opacity-50"
                                                    title={`Remove @${fn.name}`}
                                                >
                                                    {isDeletingParams.funcName === fn.name && isDeletingParams.funcType === 'poisoning' ? (
                                                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                    ) : (
                                                        <Trash2 className="w-3.5 h-3.5" />
                                                    )}
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
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
                            <div className="flex items-center gap-2">
                                <div className="relative flex-1 min-w-0" ref={dropdownRef}>
                                    <button
                                        type="button"
                                        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                                        className="w-full text-left px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 flex justify-between items-center min-w-0"
                                    >
                                        <span className="truncate flex-1 min-w-0 pr-4">{getCurrentAggregationLabel()}</span>
                                        <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
                                    </button>

                                    {isDropdownOpen && (
                                        <div className="absolute z-40 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-xl max-h-56 overflow-y-auto sidebar-scroll">
                                            {predefinedAggregationOptions.map(opt => (
                                                <button
                                                    key={opt.value}
                                                    type="button"
                                                    onClick={() => {
                                                        handleChange('data_poison_protection', opt.value);
                                                        setIsDropdownOpen(false);
                                                    }}
                                                    className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${config.data_poison_protection === opt.value ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 font-medium' : 'text-gray-700 dark:text-gray-200'}`}
                                                >
                                                    {opt.label}
                                                </button>
                                            ))}

                                            {customFunctions.length > 0 && (
                                                <>
                                                    <div className="px-3 py-1.5 text-xs font-semibold text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-900 border-t border-b border-gray-200 dark:border-gray-700 uppercase tracking-wider">
                                                        ── Custom Functions ──
                                                    </div>
                                                    {customFunctions.map(fn => (
                                                        <button
                                                            key={fn.name}
                                                            type="button"
                                                            onClick={() => {
                                                                handleChange('data_poison_protection', `@${fn.name}`);
                                                                setIsDropdownOpen(false);
                                                            }}
                                                            className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2 ${config.data_poison_protection === `@${fn.name}` ? 'bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 font-medium' : 'text-gray-700 dark:text-gray-200'}`}
                                                        >
                                                            <span>🧩</span> @{fn.name} (custom)
                                                        </button>
                                                    ))}
                                                </>
                                            )}
                                        </div>
                                    )}
                                </div>
                                <button
                                    type="button"
                                    onClick={() => setShowCustomModal(true)}
                                    className="flex-shrink-0 p-2 bg-purple-100 hover:bg-purple-200 dark:bg-purple-900/40 dark:hover:bg-purple-900/60 text-purple-700 dark:text-purple-300 rounded-lg transition-colors"
                                    title="Add custom aggregation function"
                                >
                                    <Plus className="w-5 h-5" />
                                </button>
                            </div>
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                Select the robust aggregation method to protect against data poisoning attacks
                            </p>
                        </div>

                        {/* List of custom functions with remove buttons */}
                        {customFunctions.length > 0 && (
                            <div className="mt-3">
                                <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Custom functions:</p>
                                <div className="space-y-1 max-h-36 overflow-y-auto sidebar-scroll pr-1">
                                    {customFunctions.map(fn => (
                                        <div key={fn.name} className="flex items-center justify-between px-3 py-1.5 bg-white dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600">
                                            <span className="text-sm font-mono text-purple-700 dark:text-purple-300">
                                                @{fn.name}
                                            </span>
                                            <button
                                                type="button"
                                                onClick={() => handleRemoveCustomFunction(fn.name)}
                                                disabled={isDeletingParams.funcName === fn.name && isDeletingParams.funcType === 'aggregation'}
                                                className="p-1 text-red-400 hover:text-red-600 dark:text-red-500 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors disabled:opacity-50"
                                                title={`Remove @${fn.name}`}
                                            >
                                                {isDeletingParams.funcName === fn.name && isDeletingParams.funcType === 'aggregation' ? (
                                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                ) : (
                                                    <Trash2 className="w-3.5 h-3.5" />
                                                )}
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

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

                {/* Custom Aggregation Modal */}
                {showCustomModal && (
                    <CustomAggregationModal
                        onClose={() => setShowCustomModal(false)}
                        onSave={handleAddCustomFunction}
                        apiUrl={apiUrl}
                        token={token}
                    />
                )}

                {/* Custom Poisoning Modal */}
                {showCustomPoisoningModal && (
                    <CustomPoisoningModal
                        onClose={() => setShowCustomPoisoningModal(false)}
                        onSave={handleAddCustomPoisoningFunction}
                        apiUrl={apiUrl}
                        token={token}
                    />
                )}
            </div>
        </div>
    );
}