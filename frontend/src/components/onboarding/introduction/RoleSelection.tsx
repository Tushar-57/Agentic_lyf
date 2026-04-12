import React from 'react';
import { motion } from 'framer-motion';
import { GraduationCap, Briefcase, Users, Sparkles } from 'lucide-react';
import { UserRole } from '../utils/onboardingUtils';

interface RoleSelectionProps {
  onSelect: (role: UserRole) => void;
}

const RoleSelection: React.FC<RoleSelectionProps> = ({ onSelect }) => {
  const roles = [
    {
      id: 'student',
      title: 'Student',
      icon: <GraduationCap className="w-8 h-8" />,
      description: 'Ace your studies and learn smarter',
      color: 'from-blue-500 to-indigo-600'
    },
    {
      id: 'professional',
      title: 'Professional',
      icon: <Briefcase className="w-8 h-8" />,
      description: 'Advance your career and thrive at work',
      color: 'from-purple-500 to-pink-600'
    },
    {
      id: 'freelancer',
      title: 'Freelancer',
      icon: <Users className="w-8 h-8" />,
      description: 'Juggle clients while growing your skills',
      color: 'from-emerald-500 to-teal-600'
    },
    {
      id: 'other',
      title: 'Other',
      icon: <Sparkles className="w-8 h-8" />,
      description: 'Create your own path forward',
      color: 'from-amber-500 to-orange-600'
    }
  ];

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <div className="relative">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl font-bold text-foreground mb-4">
            What's your main focus right now?
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            This shapes how I coach you.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {roles.map((role, index) => (
            <motion.button
              key={role.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => onSelect(role.id as UserRole)}
              className="group relative overflow-hidden rounded-2xl bg-card p-8 shadow-lg border border-border
                hover:shadow-xl transition-shadow duration-300"
            >
              <div className="relative z-10">
                <div className={`inline-block p-3 rounded-2xl bg-gradient-to-br ${role.color}
                  text-white mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  {role.icon}
                </div>
                
                <h3 className="text-2xl font-bold text-foreground mb-3">
                  {role.title}
                </h3>
                
                <p className="text-muted-foreground">
                  {role.description}
                </p>
              </div>
              
              <div className={`absolute inset-0 bg-gradient-to-br ${role.color}
                opacity-0 group-hover:opacity-5 transition-opacity duration-300`} />
            </motion.button>
          ))}
        </div>
      </div>
      </div>
  );
};
export default RoleSelection;