import React, { useEffect, useState } from 'react'
import { CalendarDays, Database, Loader2, Sparkles, User } from 'lucide-react'
import { ExistingProfileView } from '@/components/onboarding/ExistingProfileView'
import { OnboardingProfileEditor } from '@/components/onboarding/OnboardingProfileEditor'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { OnboardingData } from '@/components/onboarding/utils/onboardingUtils'

interface ProfileWorkspaceProps {
  onStartOnboarding: () => void
  onContinueToChat: () => void
}

export const ProfileWorkspace: React.FC<ProfileWorkspaceProps> = ({
  onStartOnboarding,
  onContinueToChat,
}) => {
  const [isLoading, setIsLoading] = useState(true)
  const [profile, setProfile] = useState<OnboardingData | null>(null)
  const [showProfileEditor, setShowProfileEditor] = useState(false)
  const [showMiniSetup, setShowMiniSetup] = useState(false)
  const [isMiniSetupSaving, setIsMiniSetupSaving] = useState(false)
  const [miniSetupError, setMiniSetupError] = useState<string | null>(null)
  const [preferencesSnapshot, setPreferencesSnapshot] = useState<{
    provider: string
    timezone: string
    checkInTime: string
    planningCadence: string
  } | null>(null)
  const [knowledgeSnapshot, setKnowledgeSnapshot] = useState<{
    totalEntries: number
    timeEntries: number
    lastUpdated: string
  } | null>(null)
  const [miniSetupForm, setMiniSetupForm] = useState({
    primaryFocus: 'productivity',
    planningCadence: 'daily',
    checkInTime: '09:00',
    financePriority: 'budgeting',
  })

  const loadProfile = async () => {
    setIsLoading(true)
    try {
      const [profileRes, preferencesRes, statsRes] = await Promise.all([
        fetch('/api/knowledge/onboarding/profile', { cache: 'no-store', headers: { 'Cache-Control': 'no-cache' } }),
        fetch('/api/knowledge/preferences', { cache: 'no-store', headers: { 'Cache-Control': 'no-cache' } }),
        fetch('/api/knowledge/stats', { cache: 'no-store', headers: { 'Cache-Control': 'no-cache' } }),
      ])

      if (profileRes.ok) {
        const data = await profileRes.json()
        setProfile(data)
      } else {
        setProfile(null)
      }

      if (preferencesRes.ok) {
        const prefData = await preferencesRes.json()
        setPreferencesSnapshot({
          provider: String(prefData?.llm_provider?.provider || 'not set'),
          timezone: String(prefData?.general?.timezone || 'not set'),
          checkInTime: String(prefData?.journal?.check_in_time || 'not set'),
          planningCadence: String(prefData?.productivity?.planning_cadence || 'not set'),
        })
      } else {
        setPreferencesSnapshot(null)
      }

      if (statsRes.ok) {
        const statsData = await statsRes.json()
        setKnowledgeSnapshot({
          totalEntries: Number(statsData?.total_entries || 0),
          timeEntries: Number(statsData?.entries_by_category?.time_entry || 0),
          lastUpdated: String(statsData?.last_updated || ''),
        })
      } else {
        setKnowledgeSnapshot(null)
      }
    } catch {
      setProfile(null)
      setPreferencesSnapshot(null)
      setKnowledgeSnapshot(null)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadProfile()
  }, [])

  const handleSaveMiniSetup = async () => {
    setIsMiniSetupSaving(true)
    setMiniSetupError(null)

    try {
      const currentRes = await fetch('/api/knowledge/preferences', {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache',
        },
      })
      if (!currentRes.ok) {
        throw new Error('Unable to load current preferences')
      }

      const currentPrefs = await currentRes.json()
      const updatedPrefs = {
        ...currentPrefs,
        productivity: {
          ...(currentPrefs.productivity || {}),
          primary_focus: miniSetupForm.primaryFocus,
          planning_cadence: miniSetupForm.planningCadence,
        },
        finance: {
          ...(currentPrefs.finance || {}),
          planning_priority: miniSetupForm.financePriority,
        },
        journal: {
          ...(currentPrefs.journal || {}),
          check_in_time: miniSetupForm.checkInTime,
        },
        general: {
          ...(currentPrefs.general || {}),
          mini_context_setup_completed: true,
          mini_context_setup_updated_at: new Date().toISOString(),
        },
      }

      const saveRes = await fetch('/api/knowledge/preferences', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
        },
        cache: 'no-store',
        body: JSON.stringify(updatedPrefs),
      })

      if (!saveRes.ok) {
        throw new Error('Unable to save mini setup preferences')
      }

      setShowMiniSetup(false)
      await loadProfile()
    } catch (error) {
      setMiniSetupError(error instanceof Error ? error.message : 'Failed to save mini setup')
    } finally {
      setIsMiniSetupSaving(false)
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-full min-h-[50vh] items-center justify-center">
        <div className="flex items-center gap-3 rounded-2xl border border-border/70 bg-white/75 px-5 py-3 text-muted-foreground shadow-sm dark:bg-slate-900/65">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading profile...
        </div>
      </div>
    )
  }

  if (!profile) {
    return (
      <div className="flex h-full min-h-[50vh] items-center justify-center p-4">
        <Card className="w-full max-w-xl border-border/70 bg-white/75 p-6 text-center shadow-sm dark:bg-slate-900/65 sm:p-8">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-teal-600 to-cyan-500 text-white shadow-md shadow-cyan-500/30">
            <User className="h-7 w-7" />
          </div>
          <h2 className="mb-2 text-2xl font-bold">No Profile Yet</h2>
          <p className="mb-6 text-sm text-muted-foreground sm:text-base">
            Complete onboarding once to create your coach profile and personalized setup.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <Button onClick={onStartOnboarding} className="gap-2" variant="gradient">
              <Sparkles className="h-4 w-4" />
              Start Onboarding
            </Button>
            <Button onClick={() => setShowMiniSetup(true)} variant="outline" className="gap-2">
              <Sparkles className="h-4 w-4" />
              Mini Context Setup
            </Button>
          </div>
        </Card>

        {showMiniSetup && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4">
            <Card className="w-full max-w-xl border-border/70 bg-white/95 p-5 shadow-2xl dark:bg-slate-900/95 sm:p-6">
              <h3 className="text-xl font-semibold">Mini Context Setup</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Optional quick setup to personalize coaching before full onboarding.
              </p>

              <div className="mt-4 grid grid-cols-1 gap-4">
                <label className="space-y-1 text-sm">
                  <span className="font-medium">Primary Focus</span>
                  <select
                    className="w-full rounded-xl border border-border/70 bg-background px-3 py-2"
                    value={miniSetupForm.primaryFocus}
                    onChange={(event) => setMiniSetupForm((prev) => ({ ...prev, primaryFocus: event.target.value }))}
                  >
                    <option value="productivity">Productivity</option>
                    <option value="health">Health</option>
                    <option value="finance">Finance</option>
                    <option value="journal">Journal</option>
                  </select>
                </label>

                <label className="space-y-1 text-sm">
                  <span className="font-medium">Planning Cadence</span>
                  <select
                    className="w-full rounded-xl border border-border/70 bg-background px-3 py-2"
                    value={miniSetupForm.planningCadence}
                    onChange={(event) => setMiniSetupForm((prev) => ({ ...prev, planningCadence: event.target.value }))}
                  >
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="adaptive">Adaptive</option>
                  </select>
                </label>

                <label className="space-y-1 text-sm">
                  <span className="font-medium">Preferred Check-In Time</span>
                  <input
                    type="time"
                    className="w-full rounded-xl border border-border/70 bg-background px-3 py-2"
                    value={miniSetupForm.checkInTime}
                    onChange={(event) => setMiniSetupForm((prev) => ({ ...prev, checkInTime: event.target.value }))}
                  />
                </label>

                <label className="space-y-1 text-sm">
                  <span className="font-medium">Finance Priority</span>
                  <select
                    className="w-full rounded-xl border border-border/70 bg-background px-3 py-2"
                    value={miniSetupForm.financePriority}
                    onChange={(event) => setMiniSetupForm((prev) => ({ ...prev, financePriority: event.target.value }))}
                  >
                    <option value="budgeting">Budgeting</option>
                    <option value="saving">Saving</option>
                    <option value="debt_reduction">Debt Reduction</option>
                    <option value="investing">Investing</option>
                  </select>
                </label>
              </div>

              {miniSetupError && (
                <p className="mt-3 text-sm text-red-600 dark:text-red-300">{miniSetupError}</p>
              )}

              <div className="mt-5 flex items-center justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={() => setShowMiniSetup(false)}
                  disabled={isMiniSetupSaving}
                >
                  Cancel
                </Button>
                <Button
                  variant="gradient"
                  onClick={handleSaveMiniSetup}
                  disabled={isMiniSetupSaving}
                >
                  {isMiniSetupSaving ? 'Saving...' : 'Save Setup'}
                </Button>
              </div>
            </Card>
          </div>
        )}
      </div>
    )
  }

  return (
    <>
      <div className="space-y-3 p-3 sm:p-4">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          <Card className="border-border/70 bg-white/80 p-4 shadow-sm dark:bg-slate-900/70">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Sparkles className="h-4 w-4 text-teal-500" />
              Preference Visibility
            </div>
            <div className="mt-2 space-y-1 text-xs text-muted-foreground">
              <p>Provider: {preferencesSnapshot?.provider || 'not set'}</p>
              <p>Timezone: {preferencesSnapshot?.timezone || 'not set'}</p>
              <p>Check-In: {preferencesSnapshot?.checkInTime || 'not set'}</p>
              <p>Cadence: {preferencesSnapshot?.planningCadence || 'not set'}</p>
            </div>
          </Card>

          <Card className="border-border/70 bg-white/80 p-4 shadow-sm dark:bg-slate-900/70">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Database className="h-4 w-4 text-cyan-500" />
              Knowledge Snapshot
            </div>
            <div className="mt-2 space-y-1 text-xs text-muted-foreground">
              <p>Total Entries: {knowledgeSnapshot?.totalEntries ?? 0}</p>
              <p>Time Entries: {knowledgeSnapshot?.timeEntries ?? 0}</p>
              <p>
                Last Updated:{' '}
                {knowledgeSnapshot?.lastUpdated
                  ? new Date(knowledgeSnapshot.lastUpdated).toLocaleString()
                  : 'not available'}
              </p>
            </div>
          </Card>

          <Card className="border-border/70 bg-white/80 p-4 shadow-sm dark:bg-slate-900/70">
            <div className="flex h-full flex-col justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold">
                <CalendarDays className="h-4 w-4 text-amber-500" />
                Context Actions
              </div>
              <Button variant="outline" className="mt-3 gap-2" onClick={() => setShowMiniSetup(true)}>
                <Sparkles className="h-4 w-4" />
                Mini Context Setup
              </Button>
            </div>
          </Card>
        </div>
      </div>
      <ExistingProfileView
        profile={profile}
        onEdit={() => setShowProfileEditor(true)}
        onContinue={onContinueToChat}
      />
      <OnboardingProfileEditor
        isOpen={showProfileEditor}
        onClose={() => setShowProfileEditor(false)}
        onSave={async () => {
          await loadProfile()
          setShowProfileEditor(false)
        }}
      />

      {showMiniSetup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4">
          <Card className="w-full max-w-xl border-border/70 bg-white/95 p-5 shadow-2xl dark:bg-slate-900/95 sm:p-6">
            <h3 className="text-xl font-semibold">Mini Context Setup</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Optional quick setup to keep your current priorities synced with Agentic.
            </p>

            <div className="mt-4 grid grid-cols-1 gap-4">
              <label className="space-y-1 text-sm">
                <span className="font-medium">Primary Focus</span>
                <select
                  className="w-full rounded-xl border border-border/70 bg-background px-3 py-2"
                  value={miniSetupForm.primaryFocus}
                  onChange={(event) => setMiniSetupForm((prev) => ({ ...prev, primaryFocus: event.target.value }))}
                >
                  <option value="productivity">Productivity</option>
                  <option value="health">Health</option>
                  <option value="finance">Finance</option>
                  <option value="journal">Journal</option>
                </select>
              </label>

              <label className="space-y-1 text-sm">
                <span className="font-medium">Planning Cadence</span>
                <select
                  className="w-full rounded-xl border border-border/70 bg-background px-3 py-2"
                  value={miniSetupForm.planningCadence}
                  onChange={(event) => setMiniSetupForm((prev) => ({ ...prev, planningCadence: event.target.value }))}
                >
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="adaptive">Adaptive</option>
                </select>
              </label>

              <label className="space-y-1 text-sm">
                <span className="font-medium">Preferred Check-In Time</span>
                <input
                  type="time"
                  className="w-full rounded-xl border border-border/70 bg-background px-3 py-2"
                  value={miniSetupForm.checkInTime}
                  onChange={(event) => setMiniSetupForm((prev) => ({ ...prev, checkInTime: event.target.value }))}
                />
              </label>

              <label className="space-y-1 text-sm">
                <span className="font-medium">Finance Priority</span>
                <select
                  className="w-full rounded-xl border border-border/70 bg-background px-3 py-2"
                  value={miniSetupForm.financePriority}
                  onChange={(event) => setMiniSetupForm((prev) => ({ ...prev, financePriority: event.target.value }))}
                >
                  <option value="budgeting">Budgeting</option>
                  <option value="saving">Saving</option>
                  <option value="debt_reduction">Debt Reduction</option>
                  <option value="investing">Investing</option>
                </select>
              </label>
            </div>

            {miniSetupError && (
              <p className="mt-3 text-sm text-red-600 dark:text-red-300">{miniSetupError}</p>
            )}

            <div className="mt-5 flex items-center justify-end gap-2">
              <Button
                variant="outline"
                onClick={() => setShowMiniSetup(false)}
                disabled={isMiniSetupSaving}
              >
                Cancel
              </Button>
              <Button
                variant="gradient"
                onClick={handleSaveMiniSetup}
                disabled={isMiniSetupSaving}
              >
                {isMiniSetupSaving ? 'Saving...' : 'Save Setup'}
              </Button>
            </div>
          </Card>
        </div>
      )}
    </>
  )
}

export default ProfileWorkspace
