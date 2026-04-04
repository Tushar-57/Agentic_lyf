import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Settings, 
  X, 
  Zap, 
  Brain, 
  Wifi, 
  WifiOff,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'

interface SettingsPanelProps {
  isOpen: boolean
  onClose: () => void
  currentProvider: 'openai' | 'ollama'
  onProviderChange: (provider: 'openai' | 'ollama') => void
  providerStatus: {
    openai: { healthy: boolean; model?: string; responseTime?: number }
    ollama: { healthy: boolean; model?: string; responseTime?: number }
  }
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  isOpen,
  onClose,
  currentProvider,
  onProviderChange,
  providerStatus
}) => {
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [isTestingCurrentProvider, setIsTestingCurrentProvider] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const [ollamaEndpoint, setOllamaEndpoint] = useState('http://localhost:11434')
  const [selectedOpenAIModel, setSelectedOpenAIModel] = useState('gpt-3.5-turbo')
  const [localProviderStatus, setLocalProviderStatus] = useState(providerStatus)

  // Available OpenAI models
  const openAIModels = [
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo', description: 'Fast and efficient' },
    { value: 'gpt-3.5-turbo-16k', label: 'GPT-3.5 Turbo 16K', description: 'Longer context' },
    { value: 'gpt-4o-mini', label: 'GPT-4o Mini', description: 'Latest mini model' },
    { value: 'gpt-4o-mini-2024-07-18', label: 'GPT-4o Mini (2024-07-18)', description: 'Specific version' },
  ]

  // Update local status when props change
  useEffect(() => {
    setLocalProviderStatus(providerStatus)
  }, [providerStatus])

  // Load stored configuration on mount
  useEffect(() => {
    if (isOpen) {
      loadStoredConfig()
    }
  }, [isOpen])

  const loadStoredConfig = async () => {
    try {
      const response = await fetch('/api/config', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (response.ok) {
        const config = await response.json()
        
        // Load OpenAI config (but don't show the actual API key for security)
        if (config.openai?.api_key) {
          setApiKey('••••••••••••••••') // Show placeholder if key exists
        }
        
        // Load OpenAI model selection
        if (config.openai?.model) {
          setSelectedOpenAIModel(config.openai.model)
        }
        
        // Load Ollama config
        if (config.ollama?.endpoint) {
          setOllamaEndpoint(config.ollama.endpoint)
        }
      }
    } catch (error) {
      console.error('Failed to load stored configuration:', error)
    }
  }

  const saveConfiguration = async (configType: 'openai' | 'ollama', configData: any) => {
    try {
      const payload: any = {}
      payload[configType] = configData

      const response = await fetch('/api/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })

      if (response.ok) {
        toast.success('Configuration saved', {
          description: `${configType.toUpperCase()} settings saved successfully`
        })
      }
    } catch (error) {
      console.error('Failed to save configuration:', error)
      toast.error('Failed to save configuration', {
        description: 'Please try again'
      })
    }
  }

  const handleProviderSwitch = async (provider: 'openai' | 'ollama') => {
  try {
    setIsTestingCurrentProvider(true)
    
    // Check if switching to OpenAI without API key
    if (provider === 'openai' && (!apiKey || apiKey === '••••••••••••••••')) {
      toast.error('OpenAI API key required', {
        description: 'Please enter your OpenAI API key first'
      })
      setIsTestingCurrentProvider(false)
      return
    }
    
    // Prepare the request body with configuration
    const requestBody = {
      provider,
      config: {}
    }

    // Add API key for OpenAI provider
    if (provider === 'openai' && apiKey && apiKey !== '••••••••••••••••') {
      requestBody.config = { 
        api_key: apiKey,
        model: selectedOpenAIModel 
      }
    } else if (provider === 'ollama') {
      requestBody.config = { endpoint: ollamaEndpoint }
    }

    // Call backend API to switch provider
    const response = await fetch('/api/llm/switch-provider', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    })

    const result = await response.json()
    
    if (response.ok && result.success) {
      onProviderChange(provider)
      toast.success(`Switched to ${provider.toUpperCase()} provider`, {
        description: result.message || `Now using ${provider === 'openai' ? 'OpenAI GPT models' : 'Local Ollama models'}`
      })
      
      // Update local provider status
      setLocalProviderStatus(prev => ({
        ...prev,
        [provider]: {
          ...prev[provider],
          healthy: true
        }
      }))
    } else {
      // Show the error message from backend
      toast.error(`Failed to switch to ${provider.toUpperCase()}`, {
        description: result.message || `Could not initialize ${provider} provider`
      })
    }
  } catch (error) {
    console.error('Provider switch error:', error)
    toast.error(`Failed to switch to ${provider.toUpperCase()}`, {
      description: 'Backend connection not available'
    })
  } finally {
    setIsTestingCurrentProvider(false)
  }
}

  const testConnection = async (provider: 'openai' | 'ollama') => {
  try {
    setIsTestingConnection(true)
    
    const requestBody = {
      provider,
      config: {}
    }

    // Add configuration based on provider
    if (provider === 'openai') {
      if (!apiKey || apiKey === '••••••••••••••••') {
        toast.error('OpenAI API key required', {
          description: 'Please enter your OpenAI API key first'
        })
        return
      }
      requestBody.config = { 
        api_key: apiKey,
        model: selectedOpenAIModel 
      }
    } else if (provider === 'ollama') {
      requestBody.config = { endpoint: ollamaEndpoint }
    }
    
    const response = await fetch(`/api/llm/test-connection`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    })

    const result = await response.json()
    
    if (result.healthy) {
      toast.success(`${provider.toUpperCase()} connection successful`, {
        description: `Response time: ${result.responseTime}ms`
      })
      
      // Update local provider status
      setLocalProviderStatus(prev => ({
        ...prev,
        [provider]: {
          healthy: true,
          model: result.model,
          responseTime: result.responseTime
        }
      }))
    } else {
      toast.error(`${provider.toUpperCase()} connection failed`, {
        description: result.error
      })
      
      // Update local provider status
      setLocalProviderStatus(prev => ({
        ...prev,
        [provider]: {
          healthy: false,
          model: prev[provider].model,
          responseTime: result.responseTime
        }
      }))
    }
  } catch (error) {
    toast.error('Connection test failed', {
      description: 'Please check your configuration'
    })
    
    // Update local provider status to show disconnected
    setLocalProviderStatus(prev => ({
      ...prev,
      [provider]: {
        healthy: false,
        model: prev[provider].model,
        responseTime: 0
      }
    }))
  } finally {
    setIsTestingConnection(false)
  }
}

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 z-50 h-full w-full max-w-md overflow-y-auto border-l border-border bg-background shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border p-4 sm:p-6">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <Settings className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold">Settings</h2>
                  <p className="text-sm text-muted-foreground">Configure your AI providers</p>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Content */}
            <div className="space-y-6 p-4 sm:p-6">
              {/* LLM Provider Selection */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    LLM Provider
                  </CardTitle>
                  <CardDescription>
                    Choose between OpenAI's cloud models or local Ollama models
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* OpenAI Option */}
                  <div className={cn(
                    "p-4 rounded-lg border-2 transition-all cursor-pointer",
                    currentProvider === 'openai' 
                      ? "border-primary bg-primary/5" 
                      : "border-border hover:border-primary/50"
                  )}
                  onClick={() => handleProviderSwitch('openai')}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
                          <Zap className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <h3 className="font-medium">OpenAI</h3>
                          <p className="text-sm text-muted-foreground">GPT-4, GPT-3.5 Turbo</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {localProviderStatus.openai.healthy ? (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-500" />
                        )}
                        <Switch 
                          checked={currentProvider === 'openai'} 
                          disabled={isTestingCurrentProvider}
                        />
                      </div>
                    </div>
                    
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-4 pt-4 border-t border-border"
                    >
                      <div className="space-y-3">
                        <Input
                          placeholder="Enter OpenAI API Key"
                          type="password"
                          value={apiKey}
                          onChange={(e) => setApiKey(e.target.value)}
                          icon={<Zap className="w-4 h-4" />}
                        />
                        
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-muted-foreground">Model</label>
                          <Select value={selectedOpenAIModel} onValueChange={setSelectedOpenAIModel}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select OpenAI model" />
                            </SelectTrigger>
                            <SelectContent>
                              {openAIModels.map((model) => (
                                <SelectItem key={model.value} value={model.value}>
                                  <div className="flex flex-col">
                                    <span className="font-medium">{model.label}</span>
                                    <span className="text-xs text-muted-foreground">{model.description}</span>
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <div className="flex gap-2">
                          <Button 
                            variant="outline" 
                            size="sm" 
                            onClick={() => testConnection('openai')}
                            disabled={isTestingConnection || !apiKey || apiKey === '••••••••••••••••'}
                            className="flex-1"
                          >
                            {isTestingConnection ? (
                              <Loader2 className="w-3 h-3 animate-spin mr-2" />
                            ) : (
                              <Wifi className="w-3 h-3 mr-2" />
                            )}
                            Test Connection
                          </Button>
                        </div>
                        {currentProvider === 'openai' && localProviderStatus.openai.healthy && (
                          <div className="text-xs text-muted-foreground">
                            Model: {localProviderStatus.openai.model} • 
                            Response: {localProviderStatus.openai.responseTime}ms
                          </div>
                        )}
                      </div>
                    </motion.div>
                  </div>

                  {/* Ollama Option */}
                  <div className={cn(
                    "p-4 rounded-lg border-2 transition-all cursor-pointer",
                    currentProvider === 'ollama' 
                      ? "border-primary bg-primary/5" 
                      : "border-border hover:border-primary/50"
                  )}
                  onClick={() => handleProviderSwitch('ollama')}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
                          <Brain className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <h3 className="font-medium">Ollama</h3>
                          <p className="text-sm text-muted-foreground">Local LLaMA, Mistral models</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {localProviderStatus.ollama.healthy ? (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-500" />
                        )}
                        <Switch 
                          checked={currentProvider === 'ollama'} 
                          disabled={isTestingCurrentProvider}
                        />
                      </div>
                    </div>
                    
                    {currentProvider === 'ollama' && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="mt-4 pt-4 border-t border-border"
                      >
                        <div className="space-y-3">
                          <Input
                            placeholder="Ollama endpoint URL"
                            value={ollamaEndpoint}
                            onChange={(e) => setOllamaEndpoint(e.target.value)}
                            icon={<Brain className="w-4 h-4" />}
                          />
                          <div className="flex gap-2">
                            <Button 
                              variant="outline" 
                              size="sm" 
                              onClick={() => testConnection('ollama')}
                              disabled={isTestingConnection}
                              className="flex-1"
                            >
                              {isTestingConnection ? (
                                <Loader2 className="w-3 h-3 animate-spin mr-2" />
                              ) : (
                                <Wifi className="w-3 h-3 mr-2" />
                              )}
                              Test Connection
                            </Button>
                          </div>
                          {providerStatus.ollama.healthy && (
                            <div className="text-xs text-muted-foreground">
                              Model: {localProviderStatus.ollama.model} • 
                              Response: {localProviderStatus.ollama.responseTime}ms
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Provider Status */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Wifi className="w-5 h-5" />
                    Connection Status
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <div className="flex items-center gap-2">
                        <div className={cn(
                          "w-2 h-2 rounded-full",
                          localProviderStatus.openai.healthy ? "bg-green-500" : "bg-red-500"
                        )} />
                        <span className="text-sm font-medium">OpenAI</span>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {localProviderStatus.openai.healthy ? 'Connected' : 'Disconnected'}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <div className="flex items-center gap-2">
                        <div className={cn(
                          "w-2 h-2 rounded-full",
                          localProviderStatus.ollama.healthy ? "bg-green-500" : "bg-red-500"
                        )} />
                        <span className="text-sm font-medium">Ollama</span>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {localProviderStatus.ollama.healthy ? 'Connected' : 'Disconnected'}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Quick Actions */}
              <Card>
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button 
                    variant="outline" 
                    className="w-full justify-start"
                    onClick={() => testConnection(currentProvider)}
                    disabled={isTestingCurrentProvider}
                  >
                    {isTestingCurrentProvider ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Wifi className="w-4 h-4 mr-2" />
                    )}
                    Test Current Provider
                  </Button>
                  
                  <Button 
                    variant="outline" 
                    className="w-full justify-start"
                    onClick={async () => {
                      // Save current configurations
                      if (apiKey && apiKey !== '••••••••••••••••') {
                        await saveConfiguration('openai', { 
                          api_key: apiKey,
                          model: selectedOpenAIModel 
                        })
                      }
                      await saveConfiguration('ollama', { endpoint: ollamaEndpoint })
                    }}
                  >
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Save Configuration
                  </Button>
                </CardContent>
              </Card>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}