import React from 'react';
import { motion } from 'framer-motion';
import { Edit, User, Target, Calendar, Sparkles } from 'lucide-react';
import { OnboardingData } from './utils/onboardingUtils';

interface ExistingProfileViewProps {
  profile: OnboardingData;
  onEdit: () => void;
  onContinue: () => void;
}

export const ExistingProfileView: React.FC<ExistingProfileViewProps> = ({
  profile,
  onEdit,
  onContinue,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto p-6"
    >
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
            Welcome Back!
          </h2>
          <div className="flex items-center gap-2">
            {profile.coachAvatar && (
              <img
                src={profile.coachAvatar}
                alt="Your AI Coach"
                className="w-12 h-12 rounded-full"
              />
            )}
          </div>
        </div>

        <p className="text-gray-600 dark:text-gray-300 mb-8">
          You've already set up your profile. Here's a summary of your preferences:
        </p>

        <div className="space-y-6">
          {/* Role */}
          <div className="flex items-start gap-4">
            <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded-lg">
              <User className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">Role</h3>
              <p className="text-gray-600 dark:text-gray-300">{profile.role}</p>
            </div>
          </div>

          {/* Goals */}
          <div className="flex items-start gap-4">
            <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded-lg">
              <Target className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Goals</h3>
              <div className="space-y-2">
                {profile.goals.map((goal) => (
                  <div
                    key={goal.id}
                    className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg"
                  >
                    <p className="font-medium text-gray-900 dark:text-white">{goal.title}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-300">{goal.description}</p>
                    {goal.linkedPriorities && goal.linkedPriorities.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {goal.linkedPriorities.map((priorityId) => {
                          const priority = profile.answers?.find((a) => a.id === priorityId);
                          return priority ? (
                            <span
                              key={priorityId}
                              className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full"
                            >
                              {priority.answer}
                            </span>
                          ) : null;
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Schedule */}
          {profile.schedule && (
            <div className="flex items-start gap-4">
              <div className="bg-emerald-100 dark:bg-emerald-900/30 p-3 rounded-lg">
                <Calendar className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white">Schedule</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Work Hours: {profile.schedule.workHours.start} - {profile.schedule.workHours.end}
                </p>
                <p className="text-gray-600 dark:text-gray-300">
                  Check-ins: {profile.schedule.checkIn.frequency}
                </p>
              </div>
            </div>
          )}

          {/* Mentor */}
          <div className="flex items-start gap-4">
            <div className="bg-amber-100 dark:bg-amber-900/30 p-3 rounded-lg">
              <Sparkles className="w-6 h-6 text-amber-600 dark:text-amber-400" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">AI Mentor</h3>
              <p className="text-gray-600 dark:text-gray-300">
                {profile.mentor.name} ({profile.mentor.archetype}, {profile.mentor.style})
              </p>
            </div>
          </div>

          {/* Priorities */}
          {profile.answers && profile.answers.length > 0 && (
            <div className="flex items-start gap-4">
              <div className="bg-pink-100 dark:bg-pink-900/30 p-3 rounded-lg">
                <Sparkles className="w-6 h-6 text-pink-600 dark:text-pink-400" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Priorities</h3>
                <div className="flex flex-wrap gap-2">
                  {profile.answers.map((answer) => (
                    <span
                      key={answer.id}
                      className="bg-pink-100 dark:bg-pink-900/30 text-pink-800 dark:text-pink-200 px-3 py-1 rounded-full text-sm"
                    >
                      {answer.answer}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="mt-8 flex gap-4">
          <button
            onClick={onEdit}
            className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-xl font-semibold hover:shadow-lg transition-all duration-300"
          >
            <Edit className="w-5 h-5" />
            Edit Profile
          </button>
          <button
            onClick={onContinue}
            className="flex-1 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white px-6 py-3 rounded-xl font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition-all duration-300"
          >
            Continue to Chat
          </button>
        </div>
      </div>
    </motion.div>
  );
};
