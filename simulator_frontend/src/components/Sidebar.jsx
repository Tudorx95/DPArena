import React, { useState, useEffect, useRef, useCallback } from 'react';
import { File, X, ChevronDown, ChevronRight, Plus, Folder, FolderOpen, FileSpreadsheet } from 'lucide-react';

export default function Sidebar({
    isOpen,
    projects,
    activeProjectId,
    activeFileId,
    onSelectProject,
    onSelectFile,
    onDeleteProject,
    onDeleteFile,
    onCreateProject,
    onCreateFile,
    onShowMultiExport,
    onReorderFiles,
    onRenameFile,
    onMoveFile,
    width = 256,
    onResize
}) {
    const [expandedProjects, setExpandedProjects] = useState(new Set([activeProjectId]));
    const [showNewProjectInput, setShowNewProjectInput] = useState(false);
    const [newProjectName, setNewProjectName] = useState('');
    const [showNewFileInput, setShowNewFileInput] = useState(null);
    const [newFileName, setNewFileName] = useState('');

    // Drag & Drop state
    const [draggedFile, setDraggedFile] = useState(null);
    const [dragOverFile, setDragOverFile] = useState(null);
    const [dragOverProject, setDragOverProject] = useState(null);

    // Rename state
    const [renamingFileId, setRenamingFileId] = useState(null);
    const [renameValue, setRenameValue] = useState('');

    // Resize state
    const isResizing = useRef(false);
    const startX = useRef(0);
    const startWidth = useRef(0);

    const handleResizeMouseDown = useCallback((e) => {
        e.preventDefault();
        isResizing.current = true;
        startX.current = e.clientX;
        startWidth.current = width;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        const handleMouseMove = (e) => {
            if (!isResizing.current) return;
            const delta = e.clientX - startX.current;
            const newWidth = Math.min(500, Math.max(150, startWidth.current + delta));
            onResize?.(newWidth);
        };

        const handleMouseUp = () => {
            isResizing.current = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
    }, [width, onResize]);

    // Handle ESC key to cancel project/file creation or rename
    useEffect(() => {
        const handleEsc = (event) => {
            if (event.key === 'Escape') {
                if (showNewProjectInput) {
                    setShowNewProjectInput(false);
                    setNewProjectName('');
                }
                if (showNewFileInput !== null) {
                    setShowNewFileInput(null);
                    setNewFileName('');
                }
                if (renamingFileId !== null) {
                    setRenamingFileId(null);
                    setRenameValue('');
                }
            }
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [showNewProjectInput, showNewFileInput, renamingFileId]);

    if (!isOpen) return null;

    const toggleProject = (projectId) => {
        const newExpanded = new Set(expandedProjects);
        if (newExpanded.has(projectId)) {
            newExpanded.delete(projectId);
        } else {
            newExpanded.add(projectId);
        }
        setExpandedProjects(newExpanded);
    };

    const handleCreateProject = () => {
        if (newProjectName.trim()) {
            onCreateProject(newProjectName);
            setNewProjectName('');
            setShowNewProjectInput(false);
        }
    };

    const handleCreateFile = (projectId) => {
        if (newFileName.trim()) {
            onCreateFile(projectId, newFileName);
            setNewFileName('');
            setShowNewFileInput(null);
        }
    };

    // Double-click to rename
    const handleDoubleClick = (file) => {
        setRenamingFileId(file.id);
        setRenameValue(file.name);
    };

    const handleRenameSubmit = (fileId) => {
        if (renameValue.trim() && renameValue !== '') {
            onRenameFile(fileId, renameValue.trim());
        }
        setRenamingFileId(null);
        setRenameValue('');
    };

    // Drag & Drop handlers for reordering within same project
    const handleDragStart = (e, file, projectId) => {
        setDraggedFile({ file, projectId });
        e.dataTransfer.effectAllowed = 'move';
    };

    const handleDragOver = (e, file) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        setDragOverFile(file?.id || null);
    };

    const handleDragOverProject = (e, projectId) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        setDragOverProject(projectId);
    };

    const handleDrop = (e, targetFile, targetProjectId) => {
        e.preventDefault();
        e.stopPropagation();

        if (!draggedFile) return;

        const sourceFile = draggedFile.file;
        const sourceProjectId = draggedFile.projectId;

        // Case 1: Moving within same project (reordering)
        if (sourceProjectId === targetProjectId && targetFile) {
            const project = projects.find(p => p.id === targetProjectId);
            if (!project) return;

            const files = [...project.files];
            const sourceIndex = files.findIndex(f => f.id === sourceFile.id);
            const targetIndex = files.findIndex(f => f.id === targetFile.id);

            if (sourceIndex === targetIndex) {
                setDraggedFile(null);
                setDragOverFile(null);
                return;
            }

            // Reorder array
            files.splice(sourceIndex, 1);
            files.splice(targetIndex, 0, sourceFile);

            // Create reorder updates with new indices
            const updates = files.map((file, index) => ({
                file_id: file.id,
                new_order: index
            }));

            onReorderFiles(updates);
        }
        // Case 2: Moving to different project
        else if (sourceProjectId !== targetProjectId) {
            const targetProject = projects.find(p => p.id === targetProjectId);
            if (!targetProject) return;

            // Determine new order in target project
            let newOrder;
            if (targetFile) {
                // Insert at target file position
                newOrder = targetProject.files.findIndex(f => f.id === targetFile.id);
            } else {
                // Insert at end
                newOrder = targetProject.files.length;
            }

            onMoveFile(sourceFile.id, targetProjectId, newOrder);
        }

        setDraggedFile(null);
        setDragOverFile(null);
        setDragOverProject(null);
    };

    const handleDropOnProject = (e, targetProjectId) => {
        e.preventDefault();
        e.stopPropagation();

        if (!draggedFile) return;

        const sourceFile = draggedFile.file;
        const sourceProjectId = draggedFile.projectId;

        // Only allow dropping on different project
        if (sourceProjectId !== targetProjectId) {
            onMoveFile(sourceFile.id, targetProjectId, null);
        }

        setDraggedFile(null);
        setDragOverProject(null);
    };

    const handleDragEnd = () => {
        setDraggedFile(null);
        setDragOverFile(null);
        setDragOverProject(null);
    };

    return (
        <div
            className="bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col h-full relative z-20 transition-colors"
            style={{ width: `${width}px`, minWidth: '150px', maxWidth: '500px' }}
        >
            {/* Header */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                    <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300">PROJECTS</h2>
                    <button
                        onClick={() => setShowNewProjectInput(true)}
                        className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                        title="New Project"
                    >
                        <Plus className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    </button>
                </div>

                {/* Multi-Export Button */}
                <button
                    onClick={onShowMultiExport}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-gradient-to-r from-green-500 to-emerald-500 dark:from-green-600 dark:to-emerald-600 text-white rounded-lg hover:from-green-600 hover:to-emerald-600 dark:hover:from-green-700 dark:hover:to-emerald-700 transition-all shadow-sm hover:shadow-md text-sm font-medium mb-3"
                    title="Export Multiple Simulations to CSV"
                >
                    <FileSpreadsheet className="w-4 h-4" />
                    <span>Multi-Export CSV</span>
                </button>

                {/* New Project Input */}
                {showNewProjectInput && (
                    <div className="flex gap-1 mt-2 relative z-30">
                        <input
                            type="text"
                            value={newProjectName}
                            onChange={(e) => setNewProjectName(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleCreateProject()}
                            placeholder="Project name..."
                            className="flex-1 px-2 py-1 text-sm border border-blue-300 dark:border-blue-600 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                            autoFocus
                        />
                        <button
                            onClick={handleCreateProject}
                            className="px-2 py-1 bg-blue-600 dark:bg-blue-700 text-white text-xs rounded hover:bg-blue-700 dark:hover:bg-blue-600"
                        >
                            ✓
                        </button>
                        <button
                            onClick={() => {
                                setShowNewProjectInput(false);
                                setNewProjectName('');
                            }}
                            className="px-2 py-1 bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-300 text-xs rounded hover:bg-gray-300 dark:hover:bg-gray-500"
                        >
                            ✕
                        </button>
                    </div>
                )}
            </div>

            {/* Projects List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2 sidebar-scroll">
                {projects.length === 0 ? (
                    <div className="text-center py-8 text-gray-500 dark:text-gray-400 text-sm">
                        <Folder className="w-12 h-12 mx-auto mb-2 text-gray-300 dark:text-gray-600" />
                        <p>No projects yet</p>
                        <p className="text-xs mt-1">Create your first project</p>
                    </div>
                ) : (
                    projects.map(project => {
                        const isExpanded = expandedProjects.has(project.id);
                        const isActive = activeProjectId === project.id;
                        const isDragOverProject = dragOverProject === project.id;

                        return (
                            <div key={project.id} className="space-y-1">
                                {/* Project Header */}
                                <div
                                    className={`flex items-center justify-between p-2 rounded cursor-pointer group ${isActive ? 'bg-blue-50 dark:bg-blue-900/30' : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                                        } ${isDragOverProject ? 'ring-2 ring-blue-400 dark:ring-blue-500 bg-blue-50 dark:bg-blue-900/30' : ''}`}
                                    onDragOver={(e) => handleDragOverProject(e, project.id)}
                                    onDrop={(e) => handleDropOnProject(e, project.id)}
                                >
                                    <div
                                        className="flex items-center gap-2 flex-1"
                                        onClick={() => {
                                            toggleProject(project.id);
                                            onSelectProject(project.id);
                                        }}
                                    >
                                        {isExpanded ? (
                                            <ChevronDown className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                                        ) : (
                                            <ChevronRight className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                                        )}
                                        {isExpanded ? (
                                            <FolderOpen className="w-4 h-4 text-blue-500 dark:text-blue-400" />
                                        ) : (
                                            <Folder className="w-4 h-4 text-blue-500 dark:text-blue-400" />
                                        )}
                                        <span className={`text-sm truncate flex-1 ${isActive ? 'text-blue-600 dark:text-blue-400 font-medium' : 'text-gray-700 dark:text-gray-300'}`}>
                                            {project.name}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setShowNewFileInput(project.id);
                                            }}
                                            className="p-1 hover:bg-blue-100 dark:hover:bg-blue-900/50 rounded"
                                            title="New File"
                                        >
                                            <Plus className="w-3 h-3 text-blue-600 dark:text-blue-400" />
                                        </button>
                                        {projects.length > 1 && (
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    if (window.confirm(`Delete project "${project.name}"?`)) {
                                                        onDeleteProject(project.id);
                                                    }
                                                }}
                                                className="p-1 hover:bg-red-100 dark:hover:bg-red-900/50 rounded"
                                                title="Delete Project"
                                            >
                                                <X className="w-3 h-3 text-red-600 dark:text-red-400" />
                                            </button>
                                        )}
                                    </div>
                                </div>

                                {/* New File Input */}
                                {showNewFileInput === project.id && (
                                    <div className="ml-6 flex gap-1 mb-2 relative z-30">
                                        <input
                                            type="text"
                                            value={newFileName}
                                            onChange={(e) => setNewFileName(e.target.value)}
                                            onKeyPress={(e) => e.key === 'Enter' && handleCreateFile(project.id)}
                                            placeholder="file.md"
                                            className="flex-1 px-2 py-1 text-sm border border-blue-300 dark:border-blue-600 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                            autoFocus
                                        />
                                        <button
                                            onClick={() => handleCreateFile(project.id)}
                                            className="px-2 py-1 bg-blue-600 dark:bg-blue-700 text-white text-xs rounded hover:bg-blue-700 dark:hover:bg-blue-600"
                                        >
                                            ✓
                                        </button>
                                        <button
                                            onClick={() => {
                                                setShowNewFileInput(null);
                                                setNewFileName('');
                                            }}
                                            className="px-2 py-1 bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-300 text-xs rounded hover:bg-gray-300 dark:hover:bg-gray-500"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                )}

                                {/* Files List */}
                                {isExpanded && project.files && (
                                    <div className="ml-6 space-y-1">
                                        {project.files.length === 0 ? (
                                            <p className="text-xs text-gray-400 dark:text-gray-500 py-2 px-2">No files yet</p>
                                        ) : (
                                            project.files.map(file => {
                                                const isFileActive = activeFileId === file.id;
                                                const isDragOver = dragOverFile === file.id;
                                                const isBeingDragged = draggedFile?.file.id === file.id;

                                                return (
                                                    <div
                                                        key={file.id}
                                                        draggable={renamingFileId !== file.id}
                                                        onDragStart={(e) => handleDragStart(e, file, project.id)}
                                                        onDragOver={(e) => handleDragOver(e, file)}
                                                        onDrop={(e) => handleDrop(e, file, project.id)}
                                                        onDragEnd={handleDragEnd}
                                                        className={`flex items-center justify-between p-2 rounded cursor-pointer group ${isFileActive
                                                            ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'
                                                            : 'hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                                                            } ${isDragOver ? 'border-t-2 border-blue-400 dark:border-blue-500' : ''} ${isBeingDragged ? 'opacity-50' : ''
                                                            }`}
                                                        onClick={() => {
                                                            if (renamingFileId !== file.id) {
                                                                // Switch to this project if not already active
                                                                if (activeProjectId !== project.id) {
                                                                    onSelectProject(project.id);
                                                                }
                                                                onSelectFile(file.id);
                                                            }
                                                        }}
                                                        onDoubleClick={() => handleDoubleClick(file)}
                                                    >
                                                        <div className="flex items-center gap-2 flex-1 min-w-0">
                                                            <File className="w-3 h-3 flex-shrink-0" />
                                                            {renamingFileId === file.id ? (
                                                                <input
                                                                    type="text"
                                                                    value={renameValue}
                                                                    onChange={(e) => setRenameValue(e.target.value)}
                                                                    onKeyPress={(e) => {
                                                                        if (e.key === 'Enter') {
                                                                            handleRenameSubmit(file.id);
                                                                        }
                                                                    }}
                                                                    onBlur={() => handleRenameSubmit(file.id)}
                                                                    className="flex-1 px-1 py-0 text-sm border border-blue-300 dark:border-blue-600 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                                                                    autoFocus
                                                                    onClick={(e) => e.stopPropagation()}
                                                                />
                                                            ) : (
                                                                <span className="text-sm truncate">{file.name}</span>
                                                            )}
                                                        </div>
                                                        {renamingFileId !== file.id && (
                                                            <button
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    if (window.confirm(`Delete file "${file.name}"?`)) {
                                                                        onDeleteFile(file.id);
                                                                    }
                                                                }}
                                                                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 dark:hover:bg-red-900/50 rounded flex-shrink-0"
                                                                title="Delete File"
                                                            >
                                                                <X className="w-3 h-3 text-red-600 dark:text-red-400" />
                                                            </button>
                                                        )}
                                                    </div>
                                                );
                                            })
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })
                )}
            </div>

            {/* Resize Handle */}
            <div
                onMouseDown={handleResizeMouseDown}
                className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-blue-400 dark:hover:bg-blue-500 transition-colors z-30"
                style={{ marginRight: '-2px' }}
            />
        </div>
    );
}
