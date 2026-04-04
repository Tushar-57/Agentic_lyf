import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Save, 
  X, 
  User, 
  Target, 
  Calendar, 
  Sparkles,
  Plus,
  Trash2,
  Award
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'

interface Goal {
  id: string
  title: string
  description?: string
  category: string
  priority: 'Low' | 'Medium' | 'High' | 'Critical'
  milestones: string[]
  endDate?: string
  smartCriteria?: {
    specific: { checked: boolean; note: string }
    measurable: { checked: boolean; note: string }
    achievable: { checked: boolean; note: string }
    relevant: { checked: boolean; note: string }
    timeBound: { checked: boolean; note: string }
  }
}

interface Mentor {
  name: string
  archetype: string
  style: string
  avatar: string
}

interface Schedule {
  workHours: { start: string; end: string }
  dndHours: { start: string; end: string }
  checkIn: { preferredTime: string; frequency: string }
  timezone: string
}

interface OnboardingProfile {
  role: string
  goals: Goal[]
  preferences: string[]
  mentor: Mentor
  schedule: Schedule
  coachAvatar: string
}

interface OnboardingProfileEditorProps {
  isOpen: boolean
  onClose: () => void
  onSave?: (profile: OnboardingProfile) => void
}

const ROLE_OPTIONS = ['student', 'professional', 'entrepreneur', 'freelancer', 'other']
const PRIORITY_OPTIONS = ['Low', 'Medium', 'High', 'Critical']
const CATEGORY_OPTIONS = ['Career', 'Health', 'Finance', 'Personal', 'Learning', 'Other']
const COMMUNICATION_STYLES = ['Direct', 'Friendly', 'Encouraging', 'Nurturing', 'Patient', 'Challenging', 'Sarcastic Poet']
const MENTOR_ARCHETYPES = ['Guide', 'Mentor', 'Coach', 'Teacher', 'Friend']

const AVAILABLE_AVATARS = Array.from({ length: 17 }, (_, i) => `/avatars/av${i + 1}.svg`)

export const OnboardingProfileEditor: React.FC<OnboardingProfileEditorProps> = ({
  isOpen,
  onClose,
  onSave
}) => {
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [activeTab, setActiveTab] = useState('profile')
  
  const [profile, setProfile] = useState<OnboardingProfile>({
    role: 'professional',
    goals: [],
    preferences: [],
    mentor: {
      name: 'AI Coach',
      archetype: 'Guide',
      style: 'Direct',
      avatar: '/avatars/av1.svg'
    },
    schedule: {
      workHours: { start: '09:00', end: '17:00' },
      dndHours: { start: '22:00', end: '08:00' },
      checkIn: { preferredTime: '09:00', frequency: 'daily' },
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'America/New_York'
    },
    coachAvatar: '/avatars/av1.svg'
  })

  const [newPreference, setNewPreference] = useState('')

  useEffect(() => {
    if (isOpen) {
      loadProfile()
    }
  }, [isOpen])

  const loadProfile = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/knowledge/onboarding/profile')
      if (response.ok) {
        const data = await response.json()
        setProfile({
          role: data.role || 'professional',
          goals: data.goals || [],
          preferences: data.preferences || [],
          mentor: data.mentor || profile.mentor,
          schedule: data.schedule || profile.schedule,
          coachAvatar: data.coachAvatar || '/avatars/av1.svg'
        })
      }
    } catch (error) {
      console.error('Error loading profile:', error)
      toast.error('Failed to load profile')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSaveProfile = async () => {
    setIsSaving(true)
    try {
      const response = await fetch('/api/knowledge/onboarding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...profile,
          planner: {
            goals: profile.goals,
            availability: profile.schedule,
            notifications: { remindersEnabled: true },
            integrations: { calendarSync: false, taskManagementSync: false }
          }
        })
      })

      if (response.ok) {
        toast.success('Profile saved successfully!')
        onSave?.(profile)
        onClose()
      } else {
        throw new Error('Failed to save profile')
      }
    } catch (error) {
      console.error('Error saving profile:', error)
      toast.error('Failed to save profile')
    } finally {
      setIsSaving(false)
    }
  }

  const addGoal = () => {
    const newGoal: Goal = {
      id: `goal-${Date.now()}`,
      title: '',
      description: '',
      category: 'Career',
      priority: 'Medium',
      milestones: []
    }
    setProfile({ ...profile, goals: [...profile.goals, newGoal] })
  }

  const updateGoal = (goalId: string, updates: Partial<Goal>) => {
    setProfile({
      ...profile,
      goals: profile.goals.map(g => g.id === goalId ? { ...g, ...updates } : g)
    })
  }

  const deleteGoal = (goalId: string) => {
    setProfile({
      ...profile,
      goals: profile.goals.filter(g => g.id !== goalId)
    })
  }

  const addMilestone = (goalId: string) => {
    const goal = profile.goals.find(g => g.id === goalId)
    if (goal) {
      updateGoal(goalId, {
        milestones: [...goal.milestones, `Milestone ${goal.milestones.length + 1}`]
      })
    }
  }

  const updateMilestone = (goalId: string, index: number, value: string) => {
    const goal = profile.goals.find(g => g.id === goalId)
    if (goal) {
      const newMilestones = [...goal.milestones]
      newMilestones[index] = value
      updateGoal(goalId, { milestones: newMilestones })
    }
  }

  const deleteMilestone = (goalId: string, index: number) => {
    const goal = profile.goals.find(g => g.id === goalId)
    if (goal) {
      updateGoal(goalId, {
        milestones: goal.milestones.filter((_, i) => i !== index)
      })
    }
  }

  const addPreference = () => {
    if (newPreference.trim()) {
      setProfile({
        ...profile,
        preferences: [...profile.preferences, newPreference.trim()]
      })
      setNewPreference('')
    }
  }

  const deletePreference = (index: number) => {
    setProfile({
      ...profile,
      preferences: profile.preferences.filter((_, i) => i !== index)
    })
  }

  if (!isOpen) return null

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-4xl max-h-[90vh] overflow-hidden bg-background rounded-lg shadow-xl"
      >
        {/* Header */}
        <div className="flex flex-col gap-3 border-b p-4 sm:flex-row sm:items-center sm:justify-between sm:p-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <User className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Edit Onboarding Profile</h2>
              <p className="text-sm text-muted-foreground">Update your goals, preferences, and settings</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="overflow-y-auto max-h-[calc(90vh-140px)]">
          {isLoading ? (
            <div className="flex items-center justify-center p-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
          ) : (
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="w-full justify-start border-b rounded-none h-auto p-0 bg-transparent overflow-x-auto">
                <TabsTrigger value="profile" className="gap-2 shrink-0">
                  <User className="w-4 h-4" />
                  Profile
                </TabsTrigger>
                <TabsTrigger value="goals" className="gap-2 shrink-0">
                  <Target className="w-4 h-4" />
                  Goals
                </TabsTrigger>
                <TabsTrigger value="preferences" className="gap-2 shrink-0">
                  <Award className="w-4 h-4" />
                  Priorities
                </TabsTrigger>
                <TabsTrigger value="mentor" className="gap-2 shrink-0">
                  <Sparkles className="w-4 h-4" />
                  Mentor
                </TabsTrigger>
                <TabsTrigger value="schedule" className="gap-2 shrink-0">
                  <Calendar className="w-4 h-4" />
                  Schedule
                </TabsTrigger>
              </TabsList>

              {/* Profile Tab */}
              <TabsContent value="profile" className="p-6 space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="role">Role</Label>
                  <Select
                    value={profile.role}
                    onValueChange={(value) => setProfile({ ...profile, role: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ROLE_OPTIONS.map(role => (
                        <SelectItem key={role} value={role}>
                          {role.charAt(0).toUpperCase() + role.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </TabsContent>

              {/* Goals Tab */}
              <TabsContent value="goals" className="p-6 space-y-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Your Goals</h3>
                  <Button onClick={addGoal} size="sm">
                    <Plus className="w-4 h-4 mr-2" />
                    Add Goal
                  </Button>
                </div>

                {profile.goals.length === 0 ? (
                  <Card className="p-8 text-center">
                    <Target className="w-12 h-12 mx-auto mb-3 text-muted-foreground" />
                    <p className="text-muted-foreground">No goals yet. Click "Add Goal" to get started!</p>
                  </Card>
                ) : (
                  <div className="space-y-4">
                    {profile.goals.map((goal) => (
                      <Card key={goal.id} className="p-4">
                        <div className="space-y-3">
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex-1 space-y-2">
                              <Input
                                placeholder="Goal title"
                                value={goal.title}
                                onChange={(e) => updateGoal(goal.id, { title: e.target.value })}
                                className="font-semibold"
                              />
                              <Textarea
                                placeholder="Description (optional)"
                                value={goal.description || ''}
                                onChange={(e) => updateGoal(goal.id, { description: e.target.value })}
                                rows={2}
                              />
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => deleteGoal(goal.id)}
                              className="text-destructive hover:text-destructive"
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>

                          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            <div className="space-y-2">
                              <Label>Category</Label>
                              <Select
                                value={goal.category}
                                onValueChange={(value) => updateGoal(goal.id, { category: value })}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {CATEGORY_OPTIONS.map(cat => (
                                    <SelectItem key={cat} value={cat}>{cat}</SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="space-y-2">
                              <Label>Priority</Label>
                              <Select
                                value={goal.priority}
                                onValueChange={(value: any) => updateGoal(goal.id, { priority: value })}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {PRIORITY_OPTIONS.map(priority => (
                                    <SelectItem key={priority} value={priority}>{priority}</SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <Label>Milestones</Label>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => addMilestone(goal.id)}
                              >
                                <Plus className="w-3 h-3 mr-1" />
                                Add
                              </Button>
                            </div>
                            {goal.milestones.map((milestone, index) => (
                              <div key={index} className="flex items-center gap-2">
                                <Input
                                  placeholder={`Milestone ${index + 1}`}
                                  value={milestone}
                                  onChange={(e) => updateMilestone(goal.id, index, e.target.value)}
                                />
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => deleteMilestone(goal.id, index)}
                                  className="text-destructive"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>

              {/* Preferences Tab */}
              <TabsContent value="preferences" className="p-6 space-y-4">
                <div className="space-y-2">
                  <Label>Your Priorities</Label>
                  <div className="flex gap-2">
                    <Input
                      placeholder="Add a priority (e.g., Time Management, Career)"
                      value={newPreference}
                      onChange={(e) => setNewPreference(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && addPreference()}
                    />
                    <Button onClick={addPreference}>
                      <Plus className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {profile.preferences.map((pref, index) => (
                    <Badge key={index} variant="secondary" className="text-sm px-3 py-1">
                      {pref}
                      <button
                        onClick={() => deletePreference(index)}
                        className="ml-2 hover:text-destructive"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              </TabsContent>

              {/* Mentor Tab */}
              <TabsContent value="mentor" className="p-6 space-y-4">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Mentor Name</Label>
                    <Input
                      value={profile.mentor.name}
                      onChange={(e) => setProfile({
                        ...profile,
                        mentor: { ...profile.mentor, name: e.target.value }
                      })}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Archetype</Label>
                    <Select
                      value={profile.mentor.archetype}
                      onValueChange={(value) => setProfile({
                        ...profile,
                        mentor: { ...profile.mentor, archetype: value }
                      })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {MENTOR_ARCHETYPES.map(archetype => (
                          <SelectItem key={archetype} value={archetype}>{archetype}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Communication Style</Label>
                    <Select
                      value={profile.mentor.style}
                      onValueChange={(value) => setProfile({
                        ...profile,
                        mentor: { ...profile.mentor, style: value }
                      })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {COMMUNICATION_STYLES.map(style => (
                          <SelectItem key={style} value={style}>{style}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Avatar</Label>
                    <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 md:grid-cols-9">
                      {AVAILABLE_AVATARS.map((avatar) => (
                        <button
                          key={avatar}
                          onClick={() => setProfile({
                            ...profile,
                            mentor: { ...profile.mentor, avatar },
                            coachAvatar: avatar
                          })}
                          className={cn(
                            "w-12 h-12 rounded-full overflow-hidden border-2 transition-all",
                            profile.mentor.avatar === avatar
                              ? "border-blue-500 ring-2 ring-blue-200"
                              : "border-transparent hover:border-gray-300"
                          )}
                        >
                          <img src={avatar} alt="Avatar" className="w-full h-full object-cover" />
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </TabsContent>

              {/* Schedule Tab */}
              <TabsContent value="schedule" className="p-6 space-y-4">
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Work Start Time</Label>
                      <Input
                        type="time"
                        value={profile.schedule.workHours.start}
                        onChange={(e) => setProfile({
                          ...profile,
                          schedule: {
                            ...profile.schedule,
                            workHours: { ...profile.schedule.workHours, start: e.target.value }
                          }
                        })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Work End Time</Label>
                      <Input
                        type="time"
                        value={profile.schedule.workHours.end}
                        onChange={(e) => setProfile({
                          ...profile,
                          schedule: {
                            ...profile.schedule,
                            workHours: { ...profile.schedule.workHours, end: e.target.value }
                          }
                        })}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                      <Label>DND Start Time</Label>
                      <Input
                        type="time"
                        value={profile.schedule.dndHours.start}
                        onChange={(e) => setProfile({
                          ...profile,
                          schedule: {
                            ...profile.schedule,
                            dndHours: { ...profile.schedule.dndHours, start: e.target.value }
                          }
                        })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>DND End Time</Label>
                      <Input
                        type="time"
                        value={profile.schedule.dndHours.end}
                        onChange={(e) => setProfile({
                          ...profile,
                          schedule: {
                            ...profile.schedule,
                            dndHours: { ...profile.schedule.dndHours, end: e.target.value }
                          }
                        })}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Daily Check-in Time</Label>
                    <Input
                      type="time"
                      value={profile.schedule.checkIn.preferredTime}
                      onChange={(e) => setProfile({
                        ...profile,
                        schedule: {
                          ...profile.schedule,
                          checkIn: { ...profile.schedule.checkIn, preferredTime: e.target.value }
                        }
                      })}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Timezone</Label>
                    <Input
                      value={profile.schedule.timezone}
                      onChange={(e) => setProfile({
                        ...profile,
                        schedule: { ...profile.schedule, timezone: e.target.value }
                      })}
                    />
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          )}
        </div>

        {/* Footer */}
        <div className="flex flex-col-reverse gap-3 border-t bg-muted/50 p-4 sm:flex-row sm:items-center sm:justify-end sm:p-6">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSaveProfile} disabled={isSaving || isLoading}>
            {isSaving ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Changes
              </>
            )}
          </Button>
        </div>
      </motion.div>
    </motion.div>
  )
}
