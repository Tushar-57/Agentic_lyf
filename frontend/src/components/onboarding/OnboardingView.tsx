import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  GraduationCap, 
  Briefcase, 
  Users, 
  Sparkles,
  ArrowRight,
  ArrowLeft,
  Target,
  Brain,
  Check
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';
import { UserRole, Goal, Mentor, PlannerData } from './utils/onboardingUtils';

interface OnboardingViewProps {
  onComplete?: (data: OnboardingData) => void;
  className?: string;
}

export interface OnboardingData {
  role: UserRole;
  goals: Goal[];
  preferences: string[];
  mentor: Mentor;
  planner: PlannerData;
}

type OnboardingStep = 'welcome' | 'role' | 'goals' | 'preferences' | 'planner' | 'complete';

export const OnboardingView: React.FC<OnboardingViewProps> = ({ onComplete, className }) => {
  const [currentStep, setCurrentStep] = useState<OnboardingStep>('welcome');
  const [selectedRole, setSelectedRole] = useState<UserRole | null>(null);
  const [selectedGoals, setSelectedGoals] = useState<string[]>([]);
  const [selectedPreferences, setSelectedPreferences] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const roles = [
    {
      id: 'Student',
      title: 'Student',
      icon: GraduationCap,
      description: 'Focus on academic goals and learning',
      color: 'from-blue-500 to-indigo-600'
    },
    {
      id: 'Professional',
      title: 'Professional',
      icon: Briefcase,
      description: 'Enhance work performance and career',
      color: 'from-purple-500 to-pink-600'
    },
    {
      id: 'Freelancer',
      title: 'Freelancer',
      icon: Users,
      description: 'Balance projects and growth',
      color: 'from-emerald-500 to-teal-600'
    },
    {
      id: 'Other',
      title: 'Other',
      icon: Sparkles,
      description: 'Custom path for your journey',
      color: 'from-amber-500 to-orange-600'
    }
  ];

  const goalsByRole: Record<string, string[]> = {
    Student: ['Improve Study Habits', 'Manage Assignments', 'Prepare for Exams', 'Build Knowledge Base'],
    Professional: ['Time Management', 'Project Organization', 'Skill Development', 'Work-Life Balance'],
    Freelancer: ['Client Management', 'Project Scheduling', 'Income Tracking', 'Skill Marketing'],
    Other: ['Personal Growth', 'Habit Building', 'Life Organization', 'Goal Achievement']
  };

  const handleRoleSelect = (role: UserRole) => {
    setSelectedRole(role);
    setSelectedGoals([]);
    setTimeout(() => setCurrentStep('goals'), 300);
  };

  const handleGoalToggle = (goal: string) => {
    setSelectedGoals(prev =>
      prev.includes(goal) ? prev.filter(g => g !== goal) : [...prev, goal]
    );
  };

  const handlePreferenceToggle = (pref: string) => {
    setSelectedPreferences(prev =>
      prev.includes(pref) ? prev.filter(p => p !== pref) : [...prev, pref]
    );
  };

  const handleComplete = async () => {
    if (!selectedRole || selectedGoals.length === 0) {
      toast.error('Please complete all steps');
      return;
    }

    setIsSubmitting(true);

    try {
      // Transform to goals format
      const goals: Goal[] = selectedGoals.map((title, idx) => ({
        id: `goal-${idx}`,
        title,
        description: `Work on ${title.toLowerCase()}`,
        category: selectedRole,
        priority: 'Medium' as const,
        milestones: [],
        smartCriteria: {
          specific: { checked: false, note: '' },
          measurable: { checked: false, note: '' },
          achievable: { checked: false, note: '' },
          relevant: { checked: false, note: '' },
          timeBound: { checked: false, note: '' }
        }
      }));

      const plannerData: PlannerData = {
        goals,
        availability: {
          workHours: { start: '09:00', end: '17:00' },
          dndHours: { start: '22:00', end: '08:00' },
          checkIn: { preferredTime: '09:00', frequency: 'daily' },
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
        },
        notifications: {
          remindersEnabled: true
        },
        integrations: {
          calendarSync: false,
          taskManagementSync: false
        }
      };

      const mentor: Mentor = {
        archetype: 'Guide',
        style: 'Friendly',
        name: 'AI Assistant',
        avatar: '🤖'
      };

      const onboardingData: OnboardingData = {
        role: selectedRole,
        goals,
        preferences: selectedPreferences,
        mentor,
        planner: plannerData
      };

      // Send to backend
      const response = await fetch('/api/knowledge/onboarding', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(onboardingData),
      });

      if (!response.ok) {
        throw new Error('Failed to save onboarding data');
      }

      toast.success('Profile created successfully!');
      setCurrentStep('complete');
      
      if (onComplete) {
        onComplete(onboardingData);
      }
    } catch (error) {
      console.error('Error saving onboarding:', error);
      toast.error('Failed to save profile. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderStepIndicator = () => {
    const steps = ['welcome', 'role', 'goals', 'preferences', 'complete'];
    const currentIndex = steps.indexOf(currentStep);

    return (
      <div className="flex items-center justify-center gap-2 mb-8">
        {steps.slice(0, -1).map((step, index) => (
          <div key={step} className="flex items-center">
            <div
              className={cn(
                'w-2 h-2 rounded-full transition-all duration-300',
                index <= currentIndex ? 'bg-primary w-8' : 'bg-muted'
              )}
            />
            {index < steps.length - 2 && <div className="w-4 h-px bg-border mx-1" />}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className={cn('h-full flex flex-col bg-background', className)}>
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          {currentStep !== 'welcome' && currentStep !== 'complete' && renderStepIndicator()}

          <AnimatePresence mode="wait">
            {/* Welcome Step */}
            {currentStep === 'welcome' && (
              <motion.div
                key="welcome"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="text-center space-y-8"
              >
                <div className="space-y-4">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.2 }}
                    className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 text-white mx-auto"
                  >
                    <Brain className="w-12 h-12" />
                  </motion.div>
                  <h1 className="text-4xl font-bold">Welcome to AI Ecosystem</h1>
                  <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                    Let's personalize your experience to help you achieve your goals more effectively
                  </p>
                </div>

                <Button
                  size="lg"
                  onClick={() => setCurrentStep('role')}
                  className="gap-2"
                >
                  Get Started
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </motion.div>
            )}

            {/* Role Selection */}
            {currentStep === 'role' && (
              <motion.div
                key="role"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <div className="text-center space-y-2">
                  <h2 className="text-3xl font-bold">What describes you best?</h2>
                  <p className="text-muted-foreground">
                    This helps us tailor your experience
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {roles.map((role, index) => {
                    const Icon = role.icon;
                    return (
                      <motion.div
                        key={role.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <Card
                          className={cn(
                            'p-6 cursor-pointer transition-all duration-300 hover:shadow-lg',
                            selectedRole === role.id && 'ring-2 ring-primary'
                          )}
                          onClick={() => handleRoleSelect(role.id as UserRole)}
                        >
                          <div className="space-y-4">
                            <div className={cn(
                              'inline-flex p-3 rounded-xl bg-gradient-to-br text-white',
                              role.color
                            )}>
                              <Icon className="w-6 h-6" />
                            </div>
                            <div>
                              <h3 className="text-xl font-semibold mb-2">{role.title}</h3>
                              <p className="text-sm text-muted-foreground">
                                {role.description}
                              </p>
                            </div>
                          </div>
                        </Card>
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            )}

            {/* Goals Selection */}
            {currentStep === 'goals' && selectedRole && (
              <motion.div
                key="goals"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <div className="text-center space-y-2">
                  <h2 className="text-3xl font-bold">What are your main goals?</h2>
                  <p className="text-muted-foreground">
                    Select areas you want to focus on (choose at least one)
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {goalsByRole[selectedRole]?.map((goal, index) => (
                    <motion.div
                      key={goal}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <Card
                        className={cn(
                          'p-4 cursor-pointer transition-all duration-200',
                          selectedGoals.includes(goal) && 'bg-primary/5 border-primary'
                        )}
                        onClick={() => handleGoalToggle(goal)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <Target className="w-5 h-5 text-primary" />
                            <span className="font-medium">{goal}</span>
                          </div>
                          {selectedGoals.includes(goal) && (
                            <Check className="w-5 h-5 text-primary" />
                          )}
                        </div>
                      </Card>
                    </motion.div>
                  ))}
                </div>

                <div className="flex gap-3 justify-center">
                  <Button
                    variant="outline"
                    onClick={() => setCurrentStep('role')}
                    className="gap-2"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    Back
                  </Button>
                  <Button
                    onClick={() => setCurrentStep('preferences')}
                    disabled={selectedGoals.length === 0}
                    className="gap-2"
                  >
                    Continue
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                </div>
              </motion.div>
            )}

            {/* Preferences */}
            {currentStep === 'preferences' && (
              <motion.div
                key="preferences"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <div className="text-center space-y-2">
                  <h2 className="text-3xl font-bold">Customize Your Experience</h2>
                  <p className="text-muted-foreground">
                    Optional: Select features you'd like enabled
                  </p>
                </div>

                <div className="grid gap-3">
                  {['Daily Check-ins', 'Smart Reminders', 'Progress Analytics', 'Weekly Reviews'].map((pref, index) => (
                    <motion.div
                      key={pref}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <Card
                        className={cn(
                          'p-4 cursor-pointer transition-all',
                          selectedPreferences.includes(pref) && 'bg-primary/5 border-primary'
                        )}
                        onClick={() => handlePreferenceToggle(pref)}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{pref}</span>
                          {selectedPreferences.includes(pref) && (
                            <Check className="w-5 h-5 text-primary" />
                          )}
                        </div>
                      </Card>
                    </motion.div>
                  ))}
                </div>

                <div className="flex gap-3 justify-center">
                  <Button
                    variant="outline"
                    onClick={() => setCurrentStep('goals')}
                    className="gap-2"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    Back
                  </Button>
                  <Button
                    onClick={handleComplete}
                    disabled={isSubmitting}
                    className="gap-2"
                  >
                    {isSubmitting ? 'Saving...' : 'Complete Setup'}
                    <Check className="w-4 h-4" />
                  </Button>
                </div>
              </motion.div>
            )}

            {/* Complete */}
            {currentStep === 'complete' && (
              <motion.div
                key="complete"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center space-y-8"
              >
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2, type: 'spring' }}
                  className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-green-100 dark:bg-green-900/20 mx-auto"
                >
                  <Check className="w-12 h-12 text-green-600 dark:text-green-400" />
                </motion.div>

                <div className="space-y-4">
                  <h2 className="text-4xl font-bold">You're All Set!</h2>
                  <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                    Your AI assistant is ready to help you achieve your goals
                  </p>
                </div>

                <Card className="p-6 max-w-md mx-auto">
                  <div className="space-y-3 text-left">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline">{selectedRole}</Badge>
                      <span className="text-sm text-muted-foreground">Your role</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <Badge>{selectedGoals.length} Goals</Badge>
                      <span className="text-sm text-muted-foreground">Selected</span>
                    </div>
                    {selectedPreferences.length > 0 && (
                      <div className="flex items-center gap-3">
                        <Badge variant="secondary">{selectedPreferences.length} Features</Badge>
                        <span className="text-sm text-muted-foreground">Enabled</span>
                      </div>
                    )}
                  </div>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};
