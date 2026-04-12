import React, { useState, Suspense, lazy } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Database, Settings, BarChart3, Eye, Box, UserPlus } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { KnowledgeBaseViewer } from './KnowledgeBaseViewer'
import { PreferencesEditor } from './PreferencesEditor'
import { AnalyticsDashboard } from './AnalyticsDashboard'
import { ManualPreferenceAdder } from './ManualPreferenceAdder'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

// Lazy load 3D visualization to prevent Three.js from loading eagerly
const Advanced3DVisualization = lazy(() => import('./Advanced3DVisualization'))

// Loading fallback for 3D visualization
const VisualizationFallback = () => (
  <div className="flex h-[500px] items-center justify-center rounded-2xl border border-border/70 bg-muted/50">
    <div className="flex flex-col items-center gap-3">
      <Skeleton className="h-12 w-12 rounded-full" />
      <Skeleton className="h-4 w-32" />
      <p className="text-sm text-muted-foreground">Loading 3D visualization...</p>
    </div>
  </div>
)

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
    <div className={cn("flex h-full min-h-0 flex-col rounded-[28px] border border-border/70 bg-card/70 p-3 pt-4 shadow-[0_24px_70px_-52px_rgba(15,23,42,0.7)] backdrop-blur-xl sm:p-4 sm:pt-5", className)}>
      <div className="mb-4 rounded-2xl border border-border/70 bg-gradient-to-r from-primary/10 via-accent/10 to-background p-4">
        <div className="section-kicker mb-2">Your Memory</div>
        <h2 className="text-xl font-semibold text-foreground sm:text-2xl">Your Personal Knowledge Base</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          See what I know about you, explore how your memories connect, and manage your personal insights.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex min-h-0 flex-1 flex-col">
        <TabsList className="no-scrollbar sticky top-0 z-10 flex w-full justify-start gap-1 overflow-x-auto rounded-2xl border border-border/70 bg-muted/80 p-1 backdrop-blur">
          <TabsTrigger value="viewer" className="gap-2 shrink-0">
            <Eye className="w-4 h-4" />
            <span className="hidden sm:inline">My Memory</span>
            <span className="sm:hidden">Memory</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="gap-2 shrink-0">
            <BarChart3 className="w-4 h-4" />
            Insights
          </TabsTrigger>
          <TabsTrigger value="visualization" className="gap-2 shrink-0">
            <Box className="w-4 h-4" />
            <span className="hidden sm:inline">Explore</span>
            <span className="sm:hidden">Explore</span>
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
                <h2 className="mb-2 text-2xl font-bold text-foreground">Your Knowledge Map</h2>
                <p className="mb-0 text-muted-foreground">
                  See how your thoughts, goals, and memories connect.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="rounded-2xl border border-border/70 bg-card/75 p-4 shadow-sm">
                  <h3 className="text-sm font-semibold mb-1 text-foreground">Semantic Mode</h3>
                  <p className="text-sm text-muted-foreground">
                    Uses backend embedding coordinates when they are available and distinct.
                  </p>
                </div>
                <div className="rounded-2xl border border-border/70 bg-card/75 p-4 shadow-sm">
                  <h3 className="text-sm font-semibold mb-1 text-foreground">Fallback Mode</h3>
                  <p className="text-sm text-muted-foreground">
                    If coordinates are not ready, nodes are shown in a stable category-based layout so the map stays usable.
                  </p>
                </div>
                <div className="rounded-2xl border border-border/70 bg-card/75 p-4 shadow-sm">
                  <h3 className="text-sm font-semibold mb-1 text-foreground">Exploration Tools</h3>
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
                Open Your Knowledge Map
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
                <h2 className="mb-2 text-2xl font-bold">Memory Settings</h2>
                <p className="mb-0 text-muted-foreground">
                  Control how I learn about you.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Your Data</h3>
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
                            Update what I know about you
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
                          <p className="font-medium">Tell Me Something New</p>
                          <p className="text-sm text-muted-foreground">
                            Add details about your goals and habits
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
                            Download your data
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
                            Restore from backup
                          </p>
                        </div>
                      </div>
                    </button>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Privacy</h3>
                  <div className="space-y-2">
                    <button className="w-full rounded-2xl border border-border/70 bg-white/75 p-4 text-left shadow-sm transition-colors hover:bg-accent/40 dark:bg-slate-900/60">
                      <div className="flex items-center gap-3">
                        <Database className="w-5 h-5 text-primary" />
                        <div>
                          <p className="font-medium">Clear All Data</p>
                          <p className="text-sm text-muted-foreground">
                            Delete everything I know about you
                          </p>
                        </div>
                      </div>
                    </button>
                    
                    <button className="w-full rounded-2xl border border-border/70 bg-white/75 p-4 text-left shadow-sm transition-colors hover:bg-accent/40 dark:bg-slate-900/60">
                      <div className="flex items-center gap-3">
                        <Database className="w-5 h-5 text-primary" />
                        <div>
                          <p className="font-medium">Storage Settings</p>
                          <p className="text-sm text-muted-foreground">
                            Choose how long I keep your data
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
          <Suspense fallback={<VisualizationFallback />}>
            <Advanced3DVisualization
              isOpen={isEmbeddingsVisualizationOpen}
              onClose={() => setIsEmbeddingsVisualizationOpen(false)}
            />
          </Suspense>
        )}
      </AnimatePresence>
    </div>
  )
}