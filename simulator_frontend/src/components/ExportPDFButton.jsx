import React from 'react';
import { FileDown, Loader2 } from 'lucide-react';
import jsPDF from 'jspdf';

export default function ExportPDFButton({ results, fileName }) {
    const [exporting, setExporting] = React.useState(false);

    const exportToPDF = async () => {
        setExporting(true);
        try {
            const pdf = new jsPDF();
            const pageWidth = pdf.internal.pageSize.getWidth();
            const pageHeight = pdf.internal.pageSize.getHeight();
            let y = 20;

            // Helper function to check if we need a new page
            const checkNewPage = (requiredSpace) => {
                if (y + requiredSpace > pageHeight - 20) {
                    pdf.addPage();
                    y = 20;
                }
            };

            // Title
            pdf.setFontSize(18);
            pdf.setFont(undefined, 'bold');
            pdf.text('FL Simulation Results', pageWidth / 2, y, { align: 'center' });
            y += 15;

            // File name
            pdf.setFontSize(12);
            pdf.setFont(undefined, 'normal');
            pdf.text(`File: ${fileName}`, 20, y);
            y += 15;

            // ========== FL CONFIGURATION SECTION ==========
            if (results.config) {
                checkNewPage(50);

                pdf.setFillColor(219, 234, 254); // Blue background
                pdf.rect(15, y - 5, pageWidth - 30, 8, 'F');

                pdf.setFontSize(14);
                pdf.setFont(undefined, 'bold');
                pdf.setTextColor(29, 78, 216); // Blue text
                pdf.text(' Federated Learning Configuration', 20, y);
                pdf.setTextColor(0, 0, 0);
                y += 12;

                pdf.setFontSize(10);
                pdf.setFont(undefined, 'normal');

                const config = results.config;
                pdf.text(`Total Clients (N): ${config.N || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Malicious Clients (M): ${config.M || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Neural Network: ${config.NN_NAME || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Training Rounds: ${config.ROUNDS || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Poisoned Rounds (R): ${config.R || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Distribution Strategy: ${config.strategy || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Local Epochs (E): ${config.EPOCHS || 3}`, 25, y);
                y += 12;
            }

            // ========== DATA POISONING ATTACK SECTION ==========
            if (results.config) {
                checkNewPage(60);

                pdf.setFillColor(254, 226, 226); // Red background
                pdf.rect(15, y - 5, pageWidth - 30, 8, 'F');

                pdf.setFontSize(14);
                pdf.setFont(undefined, 'bold');
                pdf.setTextColor(220, 38, 38); // Red text
                pdf.text(' Data Poisoning Attack Parameters', 20, y);
                pdf.setTextColor(0, 0, 0);
                y += 12;

                pdf.setFontSize(10);
                pdf.setFont(undefined, 'normal');

                const config = results.config;

                // Poisoning Operation
                const operationMap = {
                    'noise': ' Gaussian Noise',
                    'label_flip': ' Label Flip',
                    'backdoor': ' Backdoor Trigger'
                };
                const operation = operationMap[config.poison_operation] || config.poison_operation || 'N/A';
                pdf.text(`Poisoning Operation: ${operation}`, 25, y);
                y += 6;

                // Attack Intensity
                const intensity = config.poison_intensity
                    ? `${(config.poison_intensity * 100).toFixed(1)}% (${config.poison_intensity.toFixed(2)})`
                    : 'N/A';
                pdf.text(`Attack Intensity: ${intensity}`, 25, y);
                y += 6;

                // Poisoned Data Percentage
                const percentage = config.poison_percentage
                    ? `${(config.poison_percentage * 100).toFixed(1)}% (${config.poison_percentage.toFixed(2)})`
                    : 'N/A';
                pdf.text(`Poisoned Data Percentage: ${percentage}`, 25, y);
                y += 10;

                // Attack Summary
                if (config.poison_operation) {
                    pdf.setFont(undefined, 'italic');
                    pdf.setFontSize(9);
                    const summaryText = `Attack Summary: Using ${config.poison_operation} with ${intensity} intensity on ${percentage} of data for ${config.R || 0} rounds with ${config.M || 0} malicious clients.`;
                    const summaryLines = pdf.splitTextToSize(summaryText, pageWidth - 50);
                    pdf.text(summaryLines, 25, y);
                    y += summaryLines.length * 4 + 8;
                    pdf.setFont(undefined, 'normal');
                    pdf.setFontSize(10);
                }
            }

            // ========== SIMULATION RESULTS SECTION ==========
            checkNewPage(40);

            pdf.setFillColor(220, 252, 231); // Green background
            pdf.rect(15, y - 5, pageWidth - 30, 8, 'F');

            pdf.setFontSize(14);
            pdf.setFont(undefined, 'bold');
            pdf.setTextColor(22, 163, 74); // Green text
            pdf.text(' Simulation Results', 20, y);
            pdf.setTextColor(0, 0, 0);
            y += 12;

            if (results.analysis) {
                pdf.setFontSize(10);
                pdf.setFont(undefined, 'normal');

                pdf.text(`Init Accuracy: ${results.analysis.init_accuracy?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Clean Accuracy: ${results.analysis.clean_accuracy?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Clean + DP Protection Accuracy: ${results.analysis.clean_dp_accuracy?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Poisoned Accuracy: ${results.analysis.poisoned_accuracy?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Poisoned + DP Protection Accuracy: ${results.analysis.poisoned_dp_accuracy?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Drop (Clean - Poisoned): ${results.analysis.accuracy_drop?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Drop (Clean - Init): ${results.analysis.drop_clean_init?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Drop (Clean DP - Init): ${results.analysis.drop_clean_dp_init?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Drop (Poisoned - Init): ${results.analysis.drop_poison_init?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Drop (Poisoned DP - Init): ${results.analysis.drop_poison_dp_init?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`DP Protection Method: ${results.analysis.data_poison_protection_method || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`GPU Used: ${results.analysis.gpu_used || 'N/A'}`, 25, y);
                y += 12;

                // Confusion Matrix Metrics
                checkNewPage(60);
                pdf.setFont(undefined, 'bold');
                pdf.text('Confusion Matrix Metrics (Weighted Avg):', 25, y);
                y += 8;
                pdf.setFont(undefined, 'normal');

                pdf.text(`Clean Precision: ${results.analysis.clean_precision?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Clean Recall: ${results.analysis.clean_recall?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Clean F1 Score: ${results.analysis.clean_f1?.toFixed(4) || 'N/A'}`, 25, y);
                y += 8;
                pdf.text(`Clean DP Precision: ${results.analysis.clean_dp_precision?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Clean DP Recall: ${results.analysis.clean_dp_recall?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Clean DP F1 Score: ${results.analysis.clean_dp_f1?.toFixed(4) || 'N/A'}`, 25, y);
                y += 8;
                pdf.text(`Poisoned Precision: ${results.analysis.poisoned_precision?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Poisoned Recall: ${results.analysis.poisoned_recall?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`Poisoned F1 Score: ${results.analysis.poisoned_f1?.toFixed(4) || 'N/A'}`, 25, y);
                y += 8;
                pdf.text(`DP Protection Precision: ${results.analysis.poisoned_dp_precision?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`DP Protection Recall: ${results.analysis.poisoned_dp_recall?.toFixed(4) || 'N/A'}`, 25, y);
                y += 6;
                pdf.text(`DP Protection F1 Score: ${results.analysis.poisoned_dp_f1?.toFixed(4) || 'N/A'}`, 25, y);
                y += 12;
            }

            // ========== SUMMARY SECTION ==========
            if (results.summary) {
                checkNewPage(30);

                pdf.setFillColor(243, 244, 246); // Gray background
                pdf.rect(15, y - 5, pageWidth - 30, 8, 'F');

                pdf.setFontSize(14);
                pdf.setFont(undefined, 'bold');
                pdf.text(' Summary', 20, y);
                y += 10;

                pdf.setFontSize(9);
                pdf.setFont(undefined, 'normal');
                const lines = pdf.splitTextToSize(results.summary, pageWidth - 40);

                lines.forEach(line => {
                    checkNewPage(6);
                    pdf.text(line, 20, y);
                    y += 5;
                });
            }

            // ========== FOOTER ==========
            pdf.setFontSize(8);
            pdf.setTextColor(128, 128, 128);
            pdf.text(
                `Generated on ${new Date().toLocaleString()}`,
                pageWidth / 2,
                pageHeight - 10,
                { align: 'center' }
            );

            // Save
            pdf.save(`${fileName}_results_${new Date().getTime()}.pdf`);
        } catch (error) {
            console.error('Export PDF failed:', error);
            alert('Failed to export PDF');
        } finally {
            setExporting(false);
        }
    };

    return (
        <button
            onClick={exportToPDF}
            disabled={exporting}
            className="flex items-center gap-2 px-3 py-1.5 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:bg-gray-400 transition-colors"
        >
            {exporting ? (
                <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Exporting...
                </>
            ) : (
                <>
                    <FileDown className="w-4 h-4" />
                    Export PDF
                </>
            )}
        </button>
    );
}