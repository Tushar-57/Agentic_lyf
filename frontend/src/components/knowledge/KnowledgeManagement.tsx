import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Database, Settings, BarChart3, Eye, Box, UserPlus } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { KnowledgeBaseViewer } from './KnowledgeBaseViewer'
import { PreferencesEditor } from './PreferencesEditor'
import { AnalyticsDashboard } from './AnalyticsDashboard'
import { ManualPreferenceAdder } from './ManualPreferenceAdder'
import { Advanced3DVisualization } from './Advanced3DVisualization'
import { cn } from '@/lib/utils'

interface KnowledgeManagementProps {
  className?: string
}

export const KnowledgeManagement: React.FC<KnowledgeManagementProps> = ({
  className
}) => {
  const [activeTab, setActiveTab] = useState('viewer')
  const [refreshToken, setRefreshToken] = useState(0)
  const [isPreferencesEditorOpen, setIsPreferencesEditorOpen] = useState(false)
  const [isManualPreferenceAdderOpen, setIsManualPreferenceAdderOpen] = useState(false)
  const [isEmbeddingsVisualizationOpen, setIsEmbeddingsVisualizationOpen] = useState(false)

  const handleEditPreferences = () => {
    setIsPreferencesEditorOpen(true)
  }

  const handleAddManualPreference = () => {
    setIsManualPreferenceAdderOpen(true)
  }

  const handleOpenEmbeddingsVisualization = () => {
    setIsEmbeddingsVisualizationOpen(true)
  }

  const handlePreferencesSaved = () => {
    setRefreshToken((previous) => previous + 1)
  }

  const handlePreferenceAdded = () => {
    setRefreshToken((previous) => previous + 1)
  }

  return (
    <div className={cn("flex h-full min-h-0 flex-col rounded-[28px] border border-border/70 bg-white/70 p-3 pt-4 shadow-[0_24px_70px_-52px_rgba(15,23,42,0.7)] backdrop-blur-xl dark:bg-slate-950/50 sm:p-4 sm:pt-5", className)}>
      <div className="mb-4 rounded-2xl border border-border/70 bg-gradient-to-r from-teal-50/80 via-cyan-50/70 to-amber-50/70 p-4 dark:from-teal-950/35 dark:via-slate-900/60 dark:to-amber-950/30">
        <div className="section-kicker mb-2">Knowledge Center</div>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 sm:text-2xl">Memory, Signals, and Semantic Maps</h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
          Manage profile memory, inspect knowledge quality, and explore embedding relationships from one focused workspace.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex min-h-0 flex-1 flex-col">
        <TabsList className="no-scrollbar sticky top-0 z-10 flex w-full justify-start gap-1 overflow-x-auto rounded-2xl border border-border/70 bg-white/80 p-1 backdrop-blur dark:bg-slate-900/70">
          <TabsTrigger value="viewer" className="gap-2 shrink-0">
            <Eye className="w-4 h-4" />
            <span className="hidden sm:inline">Knowledge Base</span>
            <span className="sm:hidden">Base</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="gap-2 shrink-0">
            <BarChart3 className="w-4 h-4" />
            Analytics
          </TabsTrigger>
          <TabsTrigger value="visualization" className="gap-2 shrink-0">
            <Box className="w-4 h-4" />
            <span className="hidden sm:inline">Visualization</span>
            <span className="sm:hidden">3D View</span>
          </TabsTrigger>
          <TabsTrigger value="settings" className="gap-2 shrink-0">
            <Settings className="w-4 h-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        <div className="min-h-0 flex-1 overflow-hidden">
          <TabsContent value="viewer" className="h-full min-h-0 overflow-y-auto p-2 sm:p-3">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <KnowledgeBaseViewer 
                onEditPreferences={handleEditPreferences}
                onAddPreference={handleAddManualPreference}
                refreshKey={refreshToken}
              />
            </motion.div>
          </TabsContent>

          <TabsContent value="analytics" className="h-full min-h-0 overflow-y-auto p-2 sm:p-3">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 }}
            >
              <AnalyticsDashboard />
            </motion.div>
          </TabsContent>

          <TabsContent value="visualization" className="h-full min-h-0 overflow-y-auto p-2 sm:p-3">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.15 }}
              className="space-y-6"
            >
              <div className="rounded-2xl border border-border/70 bg-card/70 p-5">
                <h2 className="mb-2 text-2xl font-bold">Knowledge Landscape</h2>
                <p className="mb-0 text-muted-foreground">
                  Explore your knowledge base in an interactive graph where each node is a memory and links indicate semantic closeness.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="rounded-2xl border border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
                  <h3 className="text-sm font-semibold mb-1">Semantic Mode</h3>
                  <p className="text-sm text-muted-foreground">
                    Uses backend embedding coordinates when they are available and distinct.
                  </p>
                </div>
                <div className="rounded-2xl border border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
                  <h3 className="text-sm font-semibold mb-1">Fallback Mode</h3>
                  <p className="text-sm text-muted-foreground">
                    If coordinates are not ready, nodes are shown in a stable category-based layout so the map stays usable.
                  </p>
                </div>
                <div className="rounded-2xl border border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
                  <h3 className="text-sm font-semibold mb-1">Exploration Tools</h3>
                  <p className="text-sm text-muted-foreground">
                    Search, filter, and compare connections in Graph and List modes with node detail drill-down.
                  </p>
                </div>
              </div>

              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Drag to rotate, scroll to zoom, and right-click to pan</li>
                <li>• Click any node to open details and related entries</li>
                <li>• Switch between Graph and List view for different analysis styles</li>
                <li>• Use category/type filters to isolate focused patterns</li>
              </ul>

              <Button 
                onClick={handleOpenEmbeddingsVisualization}
                className="w-full gap-2 md:w-auto"
                size="lg"
                variant="gradient"
              >
                <Box className="w-5 h-5" />
                Open Immersive Knowledge Landscape
              </Button>
              <p className="text-xs text-muted-foreground">
                On mobile, the landscape opens in a fullscreen overlay with dedicated controls.
              </p>
            </motion.div>
          </TabsContent>

          <TabsContent value="settings" className="h-full min-h-0 overflow-y-auto p-2 sm:p-3">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.2 }}
              className="space-y-6"
            >
              <div className="rounded-2xl border border-border/70 bg-card/70 p-5">
                <h2 className="mb-2 text-2xl font-bold">Knowledge Base Settings</h2>
                <p className="mb-0 text-muted-foreground">
                  Configure how your AI agents learn and store information about your preferences.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Data Management</h3>
                  <div className="space-y-2">
                    <button
                      onClick={handleEditPreferences}
                      className="w-full rounded-2xl border border-border/70 bg-white/75 p-4 text-left shadow-sm transition-colors hover:bg-accent/40 dark:bg-slate-900/60"
                    >
                      <div className="flex items-center gap-3">
                        <Settings className="w-5 h-5 text-primary" />
                        <div>
                          <p className="font-medium">Edit Preferences</p>
                          <p className="text-sm text-muted-foreground">
                            Modify your personal preferences and settings
                          </p>
                        </div>
                      </div>
                    </button>

                    <button
                      onClick={handleAddManualPreference}
                      className="w-full rounded-2xl border border-border/70 bg-white/75 p-4 text-left shadow-sm transition-colors hover:bg-accent/40 dark:bg-slate-900/60"
                    >
                      <div className="flex items-center gap-3">
                        <UserPlus className="w-5 h-5 text-primary" />
                        <div>
                          <p className="font-medium">Add Custom Preference</p>
                          <p className="text-sm text-muted-foreground">
                            Manually add new preferences and settings
                          </p>
                        </div>
                      </div>
                    </button>
                    
                    <button className="w-full rounded-2xl border border-border/70 bg-white/75 p-4 text-left shadow-sm transition-colors hover:bg-accent/40 dark:bg-slate-900/60">
                      <div className="flex items-center gap-3">
                        <Database className="w-5 h-5 text-primary" />
                        <div>
                          <p className="font-medium">Export Data</p>
                          <p className="text-sm text-muted-foreground">
                            Download your knowledge base and preferences
                          </p>
                        </div>
                      </div>
                    </button>
                    
                    <button className="w-full rounded-2xl border border-border/70 bg-white/75 p-4 text-left shadow-sm transition-colors hover:bg-accent/40 dark:bg-slate-900/60">
                      <div className="flex items-center gap-3">
                        <Database className="w-5 h-5 text-primary" />
                        <div>
                          <p className="font-medium">Import Data</p>
                          <p className="text-sm text-muted-foreground">
                            Import knowledge base from a backup file
                          </p>
                        </div>
                      </div>
                    </button>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Privacy & Security</h3>
                  <div className="space-y-2">
                    <button className="w-full rounded-2xl border border-border/70 bg-white/75 p-4 text-left shadow-sm transition-colors hover:bg-accent/40 dark:bg-slate-900/60">
                      <div className="flex items-center gap-3">
                        <Database className="w-5 h-5 text-primary" />
                        <div>
                          <p className="font-medium">Clear All Data</p>
                          <p className="text-sm text-muted-foreground">
                            Remove all stored preferences and interactions
                          </p>
                        </div>
                      </div>
                    </button>
                    
                    <button className="w-full rounded-2xl border border-border/70 bg-white/75 p-4 text-left shadow-sm transition-colors hover:bg-accent/40 dark:bg-slate-900/60">
                      <div className="flex items-center gap-3">
                        <Database className="w-5 h-5 text-primary" />
                        <div>
                          <p className="font-medium">Data Retention</p>
                          <p className="text-sm text-muted-foreground">
                            Configure how long data is stored
                          </p>
                        </div>
                      </div>
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          </TabsContent>
        </div>
      </Tabs>

      {/* Modals */}
      <AnimatePresence>
        {isPreferencesEditorOpen && (
          <PreferencesEditor
            isOpen={isPreferencesEditorOpen}
            onClose={() => setIsPreferencesEditorOpen(false)}
            onSave={handlePreferencesSaved}
          />
        )}
        
        {isManualPreferenceAdderOpen && (
          <ManualPreferenceAdder
            isOpen={isManualPreferenceAdderOpen}
            onClose={() => setIsManualPreferenceAdderOpen(false)}
            onPreferenceAdded={handlePreferenceAdded}
          />
        )}
        
        {isEmbeddingsVisualizationOpen && (
          <Advanced3DVisualization
            isOpen={isEmbeddingsVisualizationOpen}
            onClose={() => setIsEmbeddingsVisualizationOpen(false)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}