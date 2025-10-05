import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Message, PlannerData, UserRole, Answer, Goal, Mentor, OnboardingData } from './utils/onboardingUtils';
import { ChatContainer } from './UI/ChatContainer';
import RoleSelection from './introduction/RoleSelection';
import StepGoals from './goals/StepGoals';
import Personalization from './introduction/Personalization';
import StepPlanner from './planner/StepPlanner';
import ProgressBar from './UI/ProgressBar';
import { createMessage, loadOnboardingData } from './utils/onboardingUtils';
import StepMentor from './mentor/MentorComponent';
import LoadingScreen from './loading/LoadingScreen';
import { ExistingProfileView } from './ExistingProfileView';
import { OnboardingProfileEditor } from './OnboardingProfileEditor';
import { toast } from 'sonner';
import './Onboarding.css';

interface ChatOnboardingProps {
  onComplete: (data: {
    role: UserRole;
    goals: Goal[];
    answers: Answer[];
    mentor: Mentor;
    planner: PlannerData;
  }) => void;
}

const ChatOnboarding: React.FC<ChatOnboardingProps> = ({ onComplete }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [currentStep, setCurrentStep] = useState<'intro' | 'role' | 'personalization' | 'goals' | 'planner' | 'mentor' | 'complete'>('intro');
  const [previousStep, setPreviousStep] = useState<'intro' | 'role' | 'personalization' | 'goals' | 'planner' | 'mentor' | 'complete'>('intro');
  const [selectedRole, setSelectedRole] = useState<UserRole | null>(null);
  const [selectedGoals, setSelectedGoals] = useState<Goal[]>([]);
  const [selectedAnswers, setSelectedAnswers] = useState<Answer[]>([]);
  const [selectedMentor, setMentor] = useState<Mentor | null>(null);
  const [coachAvatar, setCoachAvatar] = useState<string>('');
  const [plannerData, setPlannerData] = useState<PlannerData>({
    goals: [],
    availability: {
      workHours: { start: '09:00', end: '17:00' },
      dndHours: { start: '22:00', end: '08:00' },
      checkIn: { preferredTime: '09:00', frequency: 'daily' },
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'America/New_York',
    },
    notifications: {
      remindersEnabled: true,
    },
    integrations: {
      calendarSync: false,
      taskManagementSync: false,
    },
  });
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  
  // New state for existing profile
  const [existingProfile, setExistingProfile] = useState<OnboardingData | null>(null);
  const [showExistingProfile, setShowExistingProfile] = useState(false);
  const [isCheckingProfile, setIsCheckingProfile] = useState(true);
  const [showProfileEditor, setShowProfileEditor] = useState(false);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check for existing profile on mount
  useEffect(() => {
    const checkProfile = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/knowledge/onboarding/profile');
        if (response.ok) {
          const data = await response.json();
          // Check if profile has essential data
          if (data.role) {
            setExistingProfile(data);
            setShowExistingProfile(true);
          }
        }
      } catch (error) {
        console.log('No existing profile found');
      } finally {
        setIsCheckingProfile(false);
      }
    };
    checkProfile();
  }, []);

  const addMessage = (content: string | React.ReactNode, sender: 'user' | 'assistant', delay = 0) => {
    return new Promise<void>((resolve) => {
      setTimeout(() => {
        setMessages((prev) => [...prev, createMessage(content, sender)]);
        resolve();
      }, delay);
    });
  };

  const simulateTyping = async (duration = 1500) => {
    setIsTyping(true);
    await new Promise((resolve) => setTimeout(resolve, duration));
    setIsTyping(false);
  };

  const handleIntroductionSelect = async () => {
    setCurrentStep('role');
    await addMessage("Let's start onboarding!", 'user');
    await simulateTyping();
    await addMessage("Let's begin by selecting your role.", 'assistant', 300);
    setPreviousStep('intro');
  };

  const handleRoleSelect = async (role: UserRole) => {
    setCurrentStep('personalization');
    setSelectedRole(role);
    await addMessage(
      <div className="flex items-center gap-2">
        <span>I'm currently a</span>
        <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full text-sm">{role}</span>
      </div>,
      'user'
    );
    await simulateTyping();
    await addMessage(
      "Great choice! Let's personalize your experience by selecting your priorities.",
      'assistant',
      300
    );
    setPreviousStep('role');
  };

  const handlePersonalizationSelect = async (answers: Answer[]) => {
    setCurrentStep('goals');
    setSelectedAnswers(answers);
    await addMessage(
      <div className="flex flex-col gap-2">
        <span>My priorities are:</span>
        {answers.map((answer) => (
          <div key={answer.id} className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full text-sm">
            <span className="font-semibold">{answer.answer}</span>: {answer.description}
          </div>
        ))}
      </div>,
      'user'
    );
    await simulateTyping();
    await addMessage(
      "Thanks for sharing! Let's identify which goals you'd like to achieve.",
      'assistant',
      300
    );
    setPreviousStep('personalization');
  };

  const handleGoalsUpdate = (goals: Goal[]) => {
    setSelectedGoals(goals);
    setPlannerData({ ...plannerData, goals });
  };

  const handleGoalsSelect = async (goals: Goal[]) => {
    setCurrentStep('planner');
    setSelectedGoals(goals);
    setPlannerData({ ...plannerData, goals });
    await addMessage(
      <div className="flex flex-col gap-2">
        <span>My goals are:</span>
        {goals.map((goal) => (
          <span key={goal.id} className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full text-sm">
            {goal.title}
          </span>
        ))}
      </div>,
      'user'
    );
    await simulateTyping();
    await addMessage(
      "Awesome goals! Let's set up your planner to achieve them.",
      'assistant',
      300
    );
    setPreviousStep('goals');
  };

  const handlePlannerSubmit = async () => {
    setCurrentStep('mentor');
    await addMessage(
      <div className="flex flex-col gap-2">
        <span>My planner is set up with {plannerData.goals.length} goals.</span>
        <div className="flex flex-wrap gap-2">
          {plannerData.goals.map((goal) => (
            <span key={goal.id} className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full text-sm">
              Goal: {goal.title} ({goal.whyItMatters || 'No reason specified'})
            </span>
          ))}
          <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full text-sm">
            Availability: {plannerData.availability.workHours.start} - {plannerData.availability.workHours.end}
          </span>
          <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full text-sm">
            Reminders: {plannerData.notifications.remindersEnabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      </div>,
      'user'
    );
    await simulateTyping();
    await addMessage(
      "Great! Now, let's meet your AI Alter Ego!",
      'assistant',
      300
    );
    setPreviousStep('planner');
  };

  const handleMentorSelect = async (selectedMentor: Mentor) => {
    setMentor(selectedMentor);
    setCoachAvatar(selectedMentor.avatar);
    await addMessage(
      <div className="flex flex-col gap-2">
        <span>My AI AlterEgo:</span>
        <span className="bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200 px-2 py-1 rounded-full text-sm">
          {selectedMentor.name} ({selectedMentor.archetype}, {selectedMentor.style})
        </span>
      </div>,
      'user'
    );

    // Construct OnboardingData
    const onboardingData: any = {
      role: selectedRole!,
      goals: plannerData.goals,
      answers: selectedAnswers,
      mentor: selectedMentor,
      preferredTone: selectedMentor.style,
      coachAvatar: selectedMentor.avatar,
      schedule: plannerData.availability,
      planner: plannerData,
      preferences: selectedAnswers.map(a => a.answer),
    };

    // Send to backend
    try {
      console.log('Sending payload:', JSON.stringify(onboardingData, null, 2));
      const response = await fetch('http://localhost:8000/api/knowledge/onboarding', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(onboardingData),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to submit onboarding data: ${response.statusText}\n${errorText}`);
      }

      const result = await response.json();
      console.log('Onboarding data submitted successfully:', result);
      toast.success('Profile created successfully!');
      setIsLoading(true);

    } catch (error) {
      console.error('Error submitting onboarding data:', error);
      toast.error('Failed to save profile. Please try again.');
      return;
    }

    setCurrentStep('complete');
    setPreviousStep('mentor');
    
    onComplete({
      role: selectedRole!,
      goals: plannerData.goals,
      answers: selectedAnswers,
      planner: plannerData,
      mentor: selectedMentor,
    });
  };

  const handleBack = async () => {
    if (currentStep === 'role') {
      await simulateTyping();
      await addMessage(
        <div className="space-y-3">
          <h2 className="text-xl font-bold">Welcome to AI Ecosystem</h2>
          <p className="text-muted-foreground">Ready to become your best self with AI!</p>
        </div>,
        'assistant'
      );
      setCurrentStep('intro');
      setPreviousStep('intro');
    } else if (currentStep === 'personalization') {
      await simulateTyping();
      await addMessage("Let's begin by selecting your role.", 'assistant', 300);
      setCurrentStep('role');
      setPreviousStep('intro');
      setSelectedAnswers([]);
    } else if (currentStep === 'goals') {
      await simulateTyping();
      await addMessage(
        "Let's personalize your experience by selecting your priorities.",
        'assistant',
        300
      );
      setCurrentStep('personalization');
      setPreviousStep('role');
      setSelectedGoals([]);
      setPlannerData({ ...plannerData, goals: [] });
    } else if (currentStep === 'planner') {
      await simulateTyping();
      await addMessage(
        "Let's identify which goals you'd like to achieve.",
        'assistant',
        300
      );
      setCurrentStep('goals');
      setPreviousStep('personalization');
    } else if (currentStep === 'mentor') {
      await simulateTyping();
      await addMessage(
        "Awesome goals! Let's set up your planner to achieve them.",
        'assistant',
        300
      );
      setCurrentStep('planner');
      setPreviousStep('goals');
      setMentor(null);
    }
  };

  useEffect(() => {
    let isMounted = true;
    const initialize = async () => {
      await simulateTyping(1000);
      if (!isMounted) return;
      await addMessage(
        <div className="space-y-3">
          <h2 className="text-xl font-bold">Welcome to AI Ecosystem</h2>
          <p className="text-muted-foreground">Ready to become your best self with AI!</p>
        </div>,
        'assistant'
      );
      setCurrentStep('intro');
      setPreviousStep('intro');
    };
    initialize();
    return () => {
      isMounted = false;
    };
  }, []);

  // Handlers for ExistingProfileView
  const handleEditProfile = () => {
    setShowProfileEditor(true);
  };

  const handleContinueWithProfile = () => {
    if (existingProfile) {
      // Load existing profile data
      const role = existingProfile.role as UserRole;
      const goals = existingProfile.goals || [];
      const answers = existingProfile.answers || [];
      const mentor = existingProfile.mentor || {
        name: 'Default Coach',
        archetype: 'Guide',
        style: 'Direct',
        avatar: ''
      };
      const planner = existingProfile.planner || plannerData;
      
      setSelectedRole(role);
      setSelectedGoals(goals);
      setSelectedAnswers(answers);
      setMentor(mentor);
      setCoachAvatar(existingProfile.coachAvatar || '');
      setPlannerData(planner);
      
      // Complete onboarding with existing data
      onComplete({
        role,
        goals,
        answers,
        mentor,
        planner
      });
    }
  };

  const handleProfileSave = async (profile: any) => {
    // Refresh the profile after saving
    try {
      const response = await fetch('http://localhost:8000/api/knowledge/onboarding/profile');
      if (response.ok) {
        const data = await response.json();
        setExistingProfile(data);
        toast.success('Profile updated successfully!');
      }
    } catch (error) {
      console.error('Error refreshing profile:', error);
    }
    setShowProfileEditor(false);
  };

  // Show loading while checking for existing profile
  if (isCheckingProfile) {
    return (
      <div className="min-h-screen w-full flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <p className="text-muted-foreground">Checking your profile...</p>
        </div>
      </div>
    );
  }

  // Show existing profile view if profile exists
  if (showExistingProfile && existingProfile) {
    return (
      <>
        <ExistingProfileView
          profile={existingProfile}
          onEdit={handleEditProfile}
          onContinue={handleContinueWithProfile}
        />
        <OnboardingProfileEditor
          isOpen={showProfileEditor}
          onClose={() => setShowProfileEditor(false)}
          onSave={handleProfileSave}
        />
      </>
    );
  }

  return (
    <div className="min-h-screen w-full flex flex-col bg-background">
      {isLoading && (
        <LoadingScreen
          onComplete={() => {
            setIsLoading(false);
          }}
        />
      )}
      <div className="p-4 z-20">
        <ProgressBar currentStep={currentStep} tone={null} />
      </div>
      <div className="flex-1 overflow-auto relative">
        <ChatContainer messages={messages} isTyping={isTyping} className="pb-32 min-h-full" coachAvatar="">
          <AnimatePresence mode="wait">
            {!isTyping && currentStep !== 'complete' && (
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className="mx-auto max-w-2xl w-full my-6 px-4"
              >
                {currentStep === 'intro' && (
                  <div className="flex justify-center">
                    <button
                      onClick={handleIntroductionSelect}
                      className="bg-gradient-to-r from-blue-500 to-purple-600 dark:from-blue-600 dark:to-purple-700 text-white px-6 py-3 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
                    >
                      Start Onboarding
                    </button>
                  </div>
                )}
                {currentStep === 'role' && <RoleSelection onSelect={handleRoleSelect} />}
                {currentStep === 'personalization' && (
                  <Personalization
                    userRole={selectedRole}
                    onSelect={handlePersonalizationSelect}
                    onBack={handleBack}
                  />
                )}
                {currentStep === 'goals' && (
                  <StepGoals
                    selectedGoals={selectedGoals}
                    userRole={selectedRole}
                    onSelect={handleGoalsSelect}
                    onUpdateGoals={handleGoalsUpdate}
                    onBack={handleBack}
                    userPriorities={selectedAnswers}
                  />
                )}
                {currentStep === 'planner' && (
                  <StepPlanner
                    plannerData={plannerData}
                    onUpdatePlanner={setPlannerData}
                    onSubmit={handlePlannerSubmit}
                    setChatHistory={setChatHistory}
                    errors={{}}
                    tone={null}
                    onBack={handleBack}
                  />
                )}
                {currentStep === 'mentor' && (
                  <StepMentor onSelect={handleMentorSelect} onBack={handleBack} />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </ChatContainer>
      </div>
    </div>
  );
};

export default ChatOnboarding;
