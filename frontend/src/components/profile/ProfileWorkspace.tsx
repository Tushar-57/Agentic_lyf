import React, { useEffect, useState } from 'react'
import { Loader2, Sparkles, User } from 'lucide-react'
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

  const loadProfile = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/knowledge/onboarding/profile')
      if (response.ok) {
        const data = await response.json()
        setProfile(data)
      } else {
        setProfile(null)
      }
    } catch {
      setProfile(null)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadProfile()
  }, [])

  if (isLoading) {
    return (
      <div className="flex h-full min-h-[50vh] items-center justify-center">
        <div className="flex items-center gap-3 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading profile...
        </div>
      </div>
    )
  }

  if (!profile) {
    return (
      <div className="flex h-full min-h-[50vh] items-center justify-center p-4">
        <Card className="w-full max-w-xl p-6 text-center sm:p-8">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
            <User className="h-7 w-7" />
          </div>
          <h2 className="mb-2 text-2xl font-bold">No Profile Yet</h2>
          <p className="mb-6 text-sm text-muted-foreground sm:text-base">
            Complete onboarding once to create your coach profile and personalized setup.
          </p>
          <Button onClick={onStartOnboarding} className="gap-2">
            <Sparkles className="h-4 w-4" />
            Start Onboarding
          </Button>
        </Card>
      </div>
    )
  }

  return (
    <>
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
    </>
  )
}

export default ProfileWorkspace
