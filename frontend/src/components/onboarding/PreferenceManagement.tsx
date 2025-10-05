import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { X, Save, Plus, Trash2 } from 'lucide-react';
import { PreferenceCategory, UserPreference } from './utils/onboardingUtils';
import { toast } from 'sonner';

const PREFERENCE_CATEGORIES: PreferenceCategory[] = [
  'Productivity',
  'Health',
  'Finance',
  'Journal',
  'AI',
  'General',
];

interface PreferenceManagementProps {
  onClose: () => void;
}

export const PreferenceManagement: React.FC<PreferenceManagementProps> = ({ onClose }) => {
  const [preferences, setPreferences] = useState<UserPreference[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<PreferenceCategory>('Productivity');
  const [isLoading, setIsLoading] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newPreference, setNewPreference] = useState({
    title: '',
    description: '',
  });

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    setIsLoading(true);
    try {
      // Try to load from backend first
      const response = await fetch('http://localhost:8000/api/knowledge/preferences');
      if (response.ok) {
        const data = await response.json();
        setPreferences(data);
      } else {
        // Load from localStorage as fallback
        const localData = localStorage.getItem('userPreferences');
        if (localData) {
          setPreferences(JSON.parse(localData));
        } else {
          // Initialize with default preferences
          initializeDefaultPreferences();
        }
      }
    } catch (error) {
      console.error('Error loading preferences:', error);
      // Load from localStorage as fallback
      const localData = localStorage.getItem('userPreferences');
      if (localData) {
        setPreferences(JSON.parse(localData));
      } else {
        initializeDefaultPreferences();
      }
    } finally {
      setIsLoading(false);
    }
  };

  const initializeDefaultPreferences = () => {
    const defaults: UserPreference[] = [
      {
        id: 'prod-1',
        category: 'Productivity',
        title: 'Task Prioritization',
        description: 'Help me prioritize tasks based on urgency and importance',
        isEnabled: true,
      },
      {
        id: 'prod-2',
        category: 'Productivity',
        title: 'Time Blocking',
        description: 'Suggest time blocks for deep work and focused sessions',
        isEnabled: true,
      },
      {
        id: 'health-1',
        category: 'Health',
        title: 'Break Reminders',
        description: 'Remind me to take regular breaks for health',
        isEnabled: true,
      },
      {
        id: 'health-2',
        category: 'Health',
        title: 'Exercise Tracking',
        description: 'Track and encourage daily exercise habits',
        isEnabled: false,
      },
      {
        id: 'finance-1',
        category: 'Finance',
        title: 'Budget Tracking',
        description: 'Help monitor spending and budget adherence',
        isEnabled: false,
      },
      {
        id: 'journal-1',
        category: 'Journal',
        title: 'Daily Reflection',
        description: 'Prompt daily journaling and reflection',
        isEnabled: true,
      },
      {
        id: 'ai-1',
        category: 'AI',
        title: 'Proactive Suggestions',
        description: 'Provide proactive suggestions based on patterns',
        isEnabled: true,
      },
      {
        id: 'general-1',
        category: 'General',
        title: 'Motivational Messages',
        description: 'Send occasional motivational messages',
        isEnabled: true,
      },
    ];
    setPreferences(defaults);
    savePreferences(defaults);
  };

  const savePreferences = async (prefs: UserPreference[]) => {
    // Save to localStorage
    localStorage.setItem('userPreferences', JSON.stringify(prefs));

    // Try to save to backend
    try {
      await fetch('http://localhost:8000/api/knowledge/preferences', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(prefs),
      });
    } catch (error) {
      console.error('Error saving to backend:', error);
    }
  };

  const togglePreference = async (prefId: string) => {
    const updated = preferences.map((p) =>
      p.id === prefId ? { ...p, isEnabled: !p.isEnabled } : p
    );
    setPreferences(updated);
    await savePreferences(updated);
    toast.success('Preference updated');
  };

  const addPreference = async () => {
    if (!newPreference.title.trim()) {
      toast.error('Please enter a title');
      return;
    }

    const newPref: UserPreference = {
      id: `${selectedCategory.toLowerCase()}-${Date.now()}`,
      category: selectedCategory,
      title: newPreference.title,
      description: newPreference.description,
      isEnabled: true,
    };

    const updated = [...preferences, newPref];
    setPreferences(updated);
    await savePreferences(updated);
    setNewPreference({ title: '', description: '' });
    setShowAddForm(false);
    toast.success('Preference added');
  };

  const deletePreference = async (prefId: string) => {
    const updated = preferences.filter((p) => p.id !== prefId);
    setPreferences(updated);
    await savePreferences(updated);
    toast.success('Preference deleted');
  };

  const filteredPreferences = preferences.filter((p) => p.category === selectedCategory);

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col"
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                Edit Preferences
              </h2>
              <p className="text-gray-600 dark:text-gray-300 mt-1">
                Customize your AI agent's understanding of your preferences
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Category Sidebar */}
          <div className="w-64 border-r border-gray-200 dark:border-gray-700 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-900">
            <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-4">
              Categories
            </h3>
            <div className="space-y-2">
              {PREFERENCE_CATEGORIES.map((category) => {
                const count = preferences.filter((p) => p.category === category).length;
                return (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                      selectedCategory === category
                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 font-semibold'
                        : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>{category}</span>
                      <span className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded-full">
                        {count}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Preferences Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                {selectedCategory} Preferences
              </h3>
              <button
                onClick={() => setShowAddForm(!showAddForm)}
                className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Plus className="w-4 h-4" />
                Add New
              </button>
            </div>

            {showAddForm && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
              >
                <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
                  Add New Preference
                </h4>
                <div className="space-y-3">
                  <input
                    type="text"
                    placeholder="Title"
                    value={newPreference.title}
                    onChange={(e) => setNewPreference({ ...newPreference, title: e.target.value })}
                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                  />
                  <textarea
                    placeholder="Description"
                    value={newPreference.description}
                    onChange={(e) => setNewPreference({ ...newPreference, description: e.target.value })}
                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                    rows={3}
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={addPreference}
                      className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
                    >
                      <Save className="w-4 h-4" />
                      Save
                    </button>
                    <button
                      onClick={() => {
                        setShowAddForm(false);
                        setNewPreference({ title: '', description: '' });
                      }}
                      className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {isLoading ? (
              <p className="text-gray-500">Loading...</p>
            ) : filteredPreferences.length === 0 ? (
              <p className="text-gray-500">No preferences in this category yet. Click "Add New" to create one.</p>
            ) : (
              <div className="space-y-3">
                {filteredPreferences.map((pref) => (
                  <div
                    key={pref.id}
                    className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-blue-300 dark:hover:border-blue-500 transition-colors"
                  >
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {pref.title}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                        {pref.description}
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={pref.isEnabled}
                          onChange={() => togglePreference(pref.id)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-600 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                      </label>
                      <button
                        onClick={() => deletePreference(pref.id)}
                        className="text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 p-2"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
};
