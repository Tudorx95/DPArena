import React, { useState, useRef } from 'react';
import { X, Check, AlertTriangle, Loader2 } from 'lucide-react';
import Editor from '@monaco-editor/react';

const TEMPLATE_CODE = `import numpy as np
import random
from PIL import Image

def custom_poison(image: Image.Image, class_names: list, current_class: str, target_class: str = None, intensity: float = 0.1, percentage: float = 0.1, **kwargs):
    """
    Custom data poisoning function.

    WARNING: Ensure this function is synchronized with the dataset format.
    If you assume specific image sizes or structures, the operation may fail 
    and halt the simulation.

    Args:
        image: Original PIL Image to be poisoned.
        class_names: List of all available classes in the dataset.
        current_class: The original class of the image.
        target_class: Target class for targeted attacks (optional).
        intensity: Attack intensity (0.0 to 1.0) defining how much the image is altered.
        percentage: The percentage parameter from the UI.
        
    Returns:
        tuple: (poisoned_image, new_class)
    """
    # Example 1: Label Flip logic
    new_class = target_class if target_class and target_class in class_names else current_class
    if new_class == current_class:
        available = [c for c in class_names if c != current_class]
        if available:
            new_class = random.choice(available)
            
    # Example 2: Add random noise based on intensity
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, 25 * intensity, img_array.shape)
    poisoned_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(poisoned_array), new_class
`;

export default function CustomPoisoningModal({ onClose, onSave, apiUrl, token }) {
    const [functionName, setFunctionName] = useState('');
    const [code, setCode] = useState(TEMPLATE_CODE);
    const [error, setError] = useState(null);
    const [isValidating, setIsValidating] = useState(false);
    const [success, setSuccess] = useState(false);
    const editorRef = useRef(null);

    const handleEditorDidMount = (editor) => {
        editorRef.current = editor;
    };

    const sanitizeName = (name) => {
        // Allow only alphanumeric + underscore, convert spaces to underscores
        return name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '').toLowerCase();
    };

    const handleValidateAndUpload = async () => {
        setError(null);
        setSuccess(false);

        // Validate function name
        const sanitized = sanitizeName(functionName);
        if (!sanitized || sanitized.length < 2) {
            setError('Please enter a valid function name (at least 2 characters, alphanumeric + underscores).');
            return;
        }

        // Basic client-side check: code must contain "def custom_poison"
        if (!code.includes('def custom_poison')) {
            setError('The code must contain a function named "custom_poison".');
            return;
        }

        setIsValidating(true);

        try {
            const response = await fetch(`${apiUrl}/api/upload-poisoning`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    function_name: sanitized,
                    code: code
                })
            });

            const data = await response.json();

            if (!response.ok || data.status === 'error') {
                setError(data.detail || data.message || 'Validation failed. Please check your code.');
                setIsValidating(false);
                return;
            }

            setSuccess(true);
            setIsValidating(false);

            // After a short delay, close and return the function name
            setTimeout(() => {
                onSave({ name: sanitized, code: code });
            }, 800);

        } catch (err) {
            setError(`Connection error: ${err.message}`);
            setIsValidating(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-[60] p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-4xl h-[90vh] flex flex-col overflow-hidden border border-gray-200 dark:border-gray-700">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20">
                    <div>
                        <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                            🧪 Define Custom Poisoning Function
                        </h2>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                            Write your custom data poisoning operation in Python. It will be validated and executed during the simulation.
                        </p>
                    </div>
                </div>

                {/* Important Warning */}
                <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 p-4 shrink-0 mx-6 mt-4 rounded-r-md">
                    <div className="flex">
                        <div className="flex-shrink-0">
                            <AlertTriangle className="h-5 w-5 text-yellow-400" aria-hidden="true" />
                        </div>
                        <div className="ml-3">
                            <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                                Critical Warning
                            </h3>
                            <div className="mt-2 text-sm text-yellow-700 dark:text-yellow-300">
                                <p>
                                    This function must be perfectly synchronized with the dataset structure (e.g., image dimensions, tensor types).
                                    If the poisoning operation encounters errors or generates incompatible output, <strong>the simulation will halt automatically</strong>.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Function name input */}
                <div className="px-6 py-3 border-b border-gray-200 dark:border-gray-700">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Function Name
                    </label>
                    <input
                        type="text"
                        value={functionName}
                        onChange={(e) => setFunctionName(e.target.value)}
                        placeholder="e.g., custom_noise_flip"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400"
                    />
                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                        This name will appear as <span className="font-mono text-purple-600 dark:text-purple-400">@{sanitizeName(functionName) || 'function_name'}</span> in the poisoning operations list.
                    </p>
                </div>

                {/* Monaco Editor */}
                <div className="flex-1 min-h-0">
                    <Editor
                        height="100%"
                        defaultLanguage="python"
                        value={code}
                        onChange={(value) => setCode(value || '')}
                        onMount={handleEditorDidMount}
                        theme="vs-dark"
                        options={{
                            minimap: { enabled: false },
                            fontSize: 14,
                            lineNumbers: 'on',
                            scrollBeyondLastLine: false,
                            automaticLayout: true,
                            tabSize: 4,
                            wordWrap: 'on',
                            padding: { top: 12 }
                        }}
                    />
                </div>

                {/* Status messages */}
                {error && (
                    <div className="mx-6 mt-3 p-3 bg-red-50 dark:bg-red-900/30 rounded-lg border border-red-200 dark:border-red-800 flex items-start gap-2">
                        <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                        <p className="text-sm text-red-700 dark:text-red-300 whitespace-pre-wrap">{error}</p>
                    </div>
                )}

                {success && (
                    <div className="mx-6 mt-3 p-3 bg-green-50 dark:bg-green-900/30 rounded-lg border border-green-200 dark:border-green-800">
                        <p className="text-sm text-green-700 dark:text-green-300 font-medium">
                            ✅ Function validated and uploaded successfully!
                        </p>
                    </div>
                )}

                {/* Action buttons */}
                <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700">
                    <button
                        onClick={onClose}
                        className="px-5 py-2.5 bg-red-100 hover:bg-red-200 dark:bg-red-900/40 dark:hover:bg-red-900/60 text-red-700 dark:text-red-300 rounded-lg font-medium transition-colors flex items-center gap-2"
                    >
                        <X className="w-4 h-4" />
                        Cancel
                    </button>

                    <button
                        onClick={handleValidateAndUpload}
                        disabled={isValidating || success}
                        className={`px-5 py-2.5 rounded-lg font-medium transition-colors flex items-center gap-2 ${isValidating || success
                            ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                            : 'bg-green-600 hover:bg-green-700 dark:bg-green-700 dark:hover:bg-green-600 text-white'
                            }`}
                    >
                        {isValidating ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Validating...
                            </>
                        ) : success ? (
                            <>
                                <Check className="w-4 h-4" />
                                Uploaded!
                            </>
                        ) : (
                            <>
                                <Check className="w-4 h-4" />
                                Validate & Upload
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}
