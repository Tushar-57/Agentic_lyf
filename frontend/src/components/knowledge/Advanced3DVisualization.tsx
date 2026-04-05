import React, { useRef, useEffect, useState, useMemo, Suspense } from 'react'
import { motion } from 'framer-motion'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import {
    OrbitControls,
    Text,
    Html,
    Line,
    Sphere,
    Sparkles,
    Stats
} from '@react-three/drei'
import * as THREE from 'three'
import {
    X,
    RotateCcw,
    Info,
    Search,
    Settings,
    Maximize2,
    Minimize2,
    Box,
    ChevronLeft,
    ChevronRight,
    AlertCircle,
    CheckCircle2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { readEmbeddingsCache, writeEmbeddingsCache } from '@/lib/embeddingsCache'
import { toast } from 'sonner'

// Color scheme helper function
const getCategoryColor = (category: string): string => {
    const categoryColors: Record<string, string> = {
        'productivity': '#4a9eff',
        'health': '#00d084',
        'finance': '#ffb347',
        'journal': '#a855f7',
        'system': '#64748b',
        'interaction': '#f472b6',
        'time_entry': '#3b82f6'
    }
    return categoryColors[category] || '#64748b'
}

const toDisplayLabel = (value: string) =>
    value
        .replace(/[_-]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/\b\w/g, (letter) => letter.toUpperCase())

const isTimeEntrySignal = (category: string, entryType: string, tags: string[] = []) => {
    const normalizedCategory = String(category || '').trim().toLowerCase()
    const normalizedType = String(entryType || '').trim().toLowerCase()
    const normalizedTags = tags.map((tag) => String(tag || '').trim().toLowerCase())

    if (normalizedCategory === 'time_entry') {
        return true
    }

    if (normalizedTags.includes('time_entry')) {
        return true
    }

    return normalizedType === 'interaction' && normalizedTags.includes('alterego_sync')
}

const normalizePointCategory = (category: string, entryType: string, tags: string[] = []) => {
    if (isTimeEntrySignal(category, entryType, tags)) {
        return 'time_entry'
    }

    const normalized = String(category || '').trim().toLowerCase()
    return normalized || 'uncategorized'
}

const normalizePointType = (entryType: string, normalizedCategory: string, tags: string[] = []) => {
    if (
        normalizedCategory === 'time_entry'
        || tags.map((tag) => String(tag || '').toLowerCase()).includes('time_entry')
    ) {
        return 'time_entry'
    }

    const normalized = String(entryType || '').trim().toLowerCase()
    return normalized || 'memory'
}

interface EmbeddingPoint {
    entry_id: string
    title: string
    content: string
    category: string
    entry_type: string
    tags: string[]
    embedding?: number[]
    position_3d: [number, number, number]
    created_at: string
    updated_at: string
    similarities?: Array<{
        target_id: string
        similarity: number
    }>
}

interface EmbeddingDetails {
    entry: {
        entry_id: string
        title: string
        content: string
        category: string
        entry_type: string
        tags: string[]
        metadata: Record<string, any>
        created_at: string
        updated_at: string
    }
    embedding_info: {
        has_embedding: boolean
        embedding_dimension: number
        embedding_preview: number[] | null
    }
    similar_entries: Array<{
        entry_id: string
        title: string
        similarity_score: number
        category: string
        entry_type: string
    }>
    statistics: {
        content_length: number
        tag_count: number
        metadata_keys: string[]
    }
}

const POSITION_EPSILON = 1e-4

function stableHash(value: string): number {
    let hash = 2166136261
    for (let i = 0; i < value.length; i += 1) {
        hash ^= value.charCodeAt(i)
        hash = Math.imul(hash, 16777619)
    }
    return hash >>> 0
}

function seededOffset(seed: string, salt: number, amplitude: number): number {
    const normalized = ((stableHash(`${seed}:${salt}`) % 10000) / 10000) - 0.5
    return normalized * amplitude * 2
}

function hasMeaningfulCoordinates(points: EmbeddingPoint[]): boolean {
    if (points.length < 2) {
        return false
    }

    const uniquePositions = new Set(
        points.map((point) =>
            point.position_3d.map((value) => value.toFixed(3)).join('|')
        )
    )

    if (uniquePositions.size <= 1) {
        return false
    }

    return points.every((point) =>
        point.position_3d.every((value) => Number.isFinite(value) && Math.abs(value) < 100000)
    )
}

function hasNonZeroEmbedding(point: EmbeddingPoint): boolean {
    if (!Array.isArray(point.embedding) || point.embedding.length === 0) {
        return false
    }

    return point.embedding.some((value) => Math.abs(value) > POSITION_EPSILON)
}

function normalizeSemanticPositions(points: EmbeddingPoint[]): EmbeddingPoint[] {
    const vectors = points.map((point) => new THREE.Vector3(...point.position_3d))
    const centroid = vectors.reduce((acc, vector) => acc.add(vector), new THREE.Vector3()).divideScalar(vectors.length)

    const centeredVectors = vectors.map((vector) => vector.clone().sub(centroid))
    const maxDistance = centeredVectors.reduce((max, vector) => Math.max(max, vector.length()), 1)
    const targetRadius = 52

    return points.map((point, index) => {
        const normalized = centeredVectors[index].clone().multiplyScalar(targetRadius / maxDistance)
        return {
            ...point,
            position_3d: [normalized.x, normalized.y, normalized.z],
        }
    })
}

function buildStableFallbackLayout(rawPoints: EmbeddingPoint[]): EmbeddingPoint[] {
    const categoryGroups: Record<string, EmbeddingPoint[]> = {}
    rawPoints.forEach((point) => {
        if (!categoryGroups[point.category]) {
            categoryGroups[point.category] = []
        }
        categoryGroups[point.category].push(point)
    })

    const processedPoints: EmbeddingPoint[] = []
    const sortedCategories = Object.keys(categoryGroups).sort((a, b) => a.localeCompare(b))
    const clusterRadius = 34

    sortedCategories.forEach((category, categoryIndex) => {
        const categoryPoints = [...categoryGroups[category]].sort((a, b) => a.entry_id.localeCompare(b.entry_id))
        const angle = (categoryIndex / Math.max(sortedCategories.length, 1)) * Math.PI * 2
        const clusterCenterX = Math.cos(angle) * clusterRadius + seededOffset(category, 1, 2.5)
        const clusterCenterZ = Math.sin(angle) * clusterRadius + seededOffset(category, 2, 2.5)
        const clusterCenterY = seededOffset(category, 3, 4)

        categoryPoints.forEach((point, pointIndex) => {
            const ring = 1 + Math.floor(pointIndex / 8)
            const ringStep = 7
            const localAngle = (
                (pointIndex % 8) / 8
            ) * Math.PI * 2 + seededOffset(point.entry_id, 4, 0.4)
            const distance = ring * ringStep + seededOffset(point.entry_id, 5, 1.5)

            const x = clusterCenterX + Math.cos(localAngle) * distance + seededOffset(point.entry_id, 6, 1.5)
            const y = clusterCenterY + seededOffset(point.entry_id, 7, 5)
            const z = clusterCenterZ + Math.sin(localAngle) * distance + seededOffset(point.entry_id, 8, 1.5)

            processedPoints.push({
                ...point,
                position_3d: [x, y, z],
            })
        })
    })

    return processedPoints
}

// 3D Node Component
const EmbeddingNode: React.FC<{
    point: EmbeddingPoint
    isSelected: boolean
    isHighlighted: boolean
    onClick: (point: EmbeddingPoint) => void
    onHover: (point: EmbeddingPoint | null) => void
    colorScheme: string
    nodeSize: number
    showLabels: boolean
}> = ({ point, isSelected, isHighlighted, onClick, onHover, colorScheme, nodeSize, showLabels }) => {
    const meshRef = useRef<THREE.Mesh>(null)
    const [hovered, setHovered] = useState(false)
    const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null)

    // Cleanup timeout on unmount
    useEffect(() => {
        return () => {
            if (hoverTimeoutRef.current) {
                clearTimeout(hoverTimeoutRef.current)
            }
        }
    }, [])

    // Stable positioning - no continuous animation that causes resets
    useFrame(() => {
        if (meshRef.current) {
            // Only animate scale for selected/highlighted/hovered states
            let targetScale = nodeSize
            if (isSelected) {
                targetScale = nodeSize * 1.8
            } else if (isHighlighted) {
                targetScale = nodeSize * 1.4
            } else if (hovered) {
                targetScale = nodeSize * 1.2
            }

            // Smooth scale transition without position changes
            meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1)
        }
    })

    const getColor = () => {
        if (isSelected) return '#ff4444'
        if (isHighlighted) return '#44ffaa'

        switch (colorScheme) {
            case 'category':
                return getCategoryColor(point.category)

            case 'type':
                const typeColors: Record<string, string> = {
                    'preference': '#4a9eff',
                    'interaction': '#00d084',
                    'time_entry': '#3b82f6',
                    'insight': '#ffb347',
                    'pattern': '#a855f7'
                }
                return typeColors[point.entry_type] || '#64748b'

            case 'age':
                const age = Date.now() - new Date(point.created_at).getTime()
                const maxAge = 30 * 24 * 60 * 60 * 1000 // 30 days
                const ratio = Math.min(age / maxAge, 1)
                return `hsl(${200 - ratio * 80}, 80%, 60%)` // Cyan to blue gradient

            default:
                return '#64748b'
        }
    }

    return (
        <group position={[point.position_3d[0], point.position_3d[1], point.position_3d[2]]}>
            {/* Invisible larger hover area for easier interaction */}
            <Sphere
                args={[1.5, 8, 8]}
                visible={false}
                onClick={() => onClick(point)}
                onPointerOver={(e) => {
                    e.stopPropagation()
                    // Clear any pending hide timeout
                    if (hoverTimeoutRef.current) {
                        clearTimeout(hoverTimeoutRef.current)
                        hoverTimeoutRef.current = null
                    }
                    if (!hovered) {
                        setHovered(true)
                        onHover(point)
                    }
                    document.body.style.cursor = 'pointer'
                }}
                onPointerOut={(e) => {
                    e.stopPropagation()
                    setHovered(false)
                    document.body.style.cursor = 'auto'
                    // Longer delay to prevent flickering when mouse moves slightly
                    hoverTimeoutRef.current = setTimeout(() => {
                        onHover(null)
                    }, 400)
                }}
                onPointerMove={(e) => {
                    e.stopPropagation()
                    // Keep tooltip visible when mouse moves within the node
                    if (hoverTimeoutRef.current) {
                        clearTimeout(hoverTimeoutRef.current)
                        hoverTimeoutRef.current = null
                    }
                    if (!hovered) {
                        setHovered(true)
                        onHover(point)
                    }
                }}
            />

            {/* Visible node */}
            <Sphere
                ref={meshRef}
                args={[0.8, 16, 16]}
            >
                <meshBasicMaterial
                    color={getColor()}
                    transparent
                    opacity={hovered || isSelected || isHighlighted ? 1.0 : 0.8}
                />
            </Sphere>

            {/* Outer glow ring for selected/highlighted nodes */}
            {(isSelected || isHighlighted) && (
                <Sphere args={[1.2, 16, 16]}>
                    <meshBasicMaterial
                        color={getColor()}
                        transparent
                        opacity={0.3}
                        side={THREE.BackSide}
                    />
                </Sphere>
            )}

            {/* Enhanced 3D Labels */}
            {showLabels && (
                <group>
                    <Text
                        position={[0, 1.8, 0]}
                        fontSize={0.8}
                        color="white"
                        anchorX="center"
                        anchorY="middle"
                        outlineWidth={0.1}
                        outlineColor="black"
                    >
                        {point.title.length > 15 ? point.title.substring(0, 15) + '...' : point.title}
                    </Text>
                    {/* Category indicator below title */}
                    <Text
                        position={[0, 1.3, 0]}
                        fontSize={0.4}
                        color={getCategoryColor(point.category)}
                        anchorX="center"
                        anchorY="middle"
                        outlineWidth={0.05}
                        outlineColor="black"
                    >
                        {toDisplayLabel(point.category).toUpperCase()}
                    </Text>
                </group>
            )}
        </group>
    )
}

// Connection Lines Component
const ConnectionLines: React.FC<{
    points: EmbeddingPoint[]
    selectedPoint: EmbeddingPoint | null
    similarityThreshold: number
    showAllConnections: boolean
}> = ({ points, selectedPoint, similarityThreshold, showAllConnections }) => {
    // Robust similarity calculation
    const calcSimilarity = (a: EmbeddingPoint, b: EmbeddingPoint) => {
        let similarity = 0
        if (a.category && b.category && a.category === b.category) similarity += 0.5
        const tagsA = Array.isArray(a.tags) ? a.tags : []
        const tagsB = Array.isArray(b.tags) ? b.tags : []
        const commonTags = tagsA.filter(tag => tagsB.includes(tag)).length
        if (tagsA.length > 0 && tagsB.length > 0) {
            similarity += (commonTags / Math.max(tagsA.length, tagsB.length)) * 0.5
        }
        return similarity
    }

    const connections = useMemo(() => {
        const lines: Array<{
            start: [number, number, number]
            end: [number, number, number]
            strength: number
            color: string
        }> = []

        if (showAllConnections) {
            // Show all connections above threshold
            for (let i = 0; i < points.length; i++) {
                for (let j = i + 1; j < points.length; j++) {
                    const point1 = points[i]
                    const point2 = points[j]
                    const similarity = calcSimilarity(point1, point2)
                    if (similarity >= similarityThreshold) {
                        lines.push({
                            start: point1.position_3d,
                            end: point2.position_3d,
                            strength: similarity,
                            color: similarity > 0.7 ? '#4ecdc4' : '#6b7280'
                        })
                    }
                }
            }
        } else if (selectedPoint) {
            // Show connections from selected point to similar points
            points.forEach(point => {
                if (point.entry_id !== selectedPoint.entry_id) {
                    const similarity = calcSimilarity(selectedPoint, point)
                    if (similarity >= similarityThreshold) {
                        lines.push({
                            start: selectedPoint.position_3d,
                            end: point.position_3d,
                            strength: similarity,
                            color: similarity > 0.7 ? '#4ecdc4' : '#6b7280'
                        })
                    }
                }
            })
        }
        return lines
    }, [points, selectedPoint?.entry_id, similarityThreshold, showAllConnections])

    return (
        <>
            {connections.map((connection, index) => (
                <Line
                    key={index}
                    points={[connection.start, connection.end]}
                    color={connection.strength > 0.7 ? '#44ffaa' : '#4a9eff'}
                    transparent
                    opacity={connection.strength > 0.7 ? 0.8 : 0.4}
                />
            ))}
        </>
    )
}

// Professional Hover Tooltip System
const HoverTooltip: React.FC<{
    point: EmbeddingPoint | null
    position: THREE.Vector3 | null
}> = ({ point, position }) => {
    const tooltipRef = useRef<THREE.Group>(null)
    const { camera } = useThree()
    const [isVisible, setIsVisible] = useState(false)

    // Enhanced tooltip positioning and sizing logic
    useFrame(() => {
        if (tooltipRef.current && position && camera) {
            // Always face the camera
            tooltipRef.current.lookAt(camera.position)

            // Calculate distance for intelligent scaling
            const distance = camera.position.distanceTo(new THREE.Vector3(position.x, position.y, position.z))

            // Much more aggressive scaling for distant nodes
            // Base scale starts at 2.0, goes up to 8.0 for very distant nodes
            const baseScale = 2.0
            const maxScale = 10.0
            const scaleFactor = Math.max(baseScale, Math.min(maxScale, (distance / 20) * baseScale))

            tooltipRef.current.scale.setScalar(scaleFactor)

            // Fade in/out animation
            if (!isVisible && point) {
                setIsVisible(true)
            }
        }
    })

    // Reset visibility when point changes
    useEffect(() => {
        if (!point) {
            setIsVisible(false)
        }
    }, [point])

    if (!point || !position) return null

    return (
        <group ref={tooltipRef} position={[position.x, position.y + 2.5, position.z]}>
            <Html
                center
                distanceFactor={30}
                zIndexRange={[1000, 0]}
                pointerEvents="none"
                transform={false}
                sprite
            >
                <div
                    className={`bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white rounded-2xl shadow-2xl border border-gray-500/50 backdrop-blur-xl pointer-events-none transition-all duration-300 ${isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
                        }`}
                    style={{
                        width: '380px',
                        padding: '24px',
                        transform: 'translate(-50%, -100%)',
                        marginBottom: '16px',
                        boxShadow: `
                            0 32px 64px -12px rgba(0, 0, 0, 0.9),
                            0 0 0 1px rgba(255, 255, 255, 0.1),
                            inset 0 1px 0 rgba(255, 255, 255, 0.1)
                        `,
                        background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,20,0.95) 100%)'
                    }}
                >
                    {/* Header with title and category indicator */}
                    <div className="flex items-start justify-between mb-3">
                        <h4 className="font-bold text-xl text-white leading-tight flex-1 pr-3">
                            {point.title}
                        </h4>
                        <div
                            className="w-4 h-4 rounded-full flex-shrink-0 mt-1"
                            style={{ backgroundColor: getCategoryColor(point.category) }}
                        />
                    </div>

                    {/* Content with better typography */}
                    <p className="text-base text-gray-200 mb-4 leading-relaxed font-medium">
                        {point.content.length > 180 ? point.content.substring(0, 180) + '...' : point.content}
                    </p>

                    {/* Enhanced badges */}
                    <div className="flex flex-wrap gap-2 mb-4">
                        <Badge
                            variant="outline"
                            className="text-sm font-semibold px-3 py-1.5 rounded-full border-2"
                            style={{
                                backgroundColor: `${getCategoryColor(point.category)}20`,
                                borderColor: `${getCategoryColor(point.category)}60`,
                                color: '#ffffff'
                            }}
                        >
                            📁 {point.category}
                        </Badge>
                        <Badge
                            variant="outline"
                            className="text-sm font-semibold bg-amber-500/30 text-amber-100 border-amber-400/60 px-3 py-1.5 rounded-full border-2"
                        >
                            🔖 {point.entry_type}
                        </Badge>
                    </div>

                    {/* Enhanced metadata */}
                    <div className="text-sm text-gray-300 space-y-2 border-t border-gray-700/50 pt-3">
                        <div className="flex items-center gap-3">
                            <span className="text-blue-400 text-base">📅</span>
                            <span className="font-medium">{new Date(point.created_at).toLocaleDateString('en-US', {
                                year: 'numeric',
                                month: 'short',
                                day: 'numeric'
                            })}</span>
                        </div>
                        <div className="flex items-start gap-3">
                            <span className="text-green-400 text-base mt-0.5">🏷️</span>
                            <div className="flex-1">
                                <div className="flex flex-wrap gap-1">
                                    {point.tags.slice(0, 5).map(tag => (
                                        <span
                                            key={tag}
                                            className="inline-block bg-gray-700/50 text-gray-300 px-2 py-0.5 rounded text-xs font-medium"
                                        >
                                            {tag}
                                        </span>
                                    ))}
                                    {point.tags.length > 5 && (
                                        <span className="inline-block bg-gray-600/50 text-gray-400 px-2 py-0.5 rounded text-xs">
                                            +{point.tags.length - 5}
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Professional arrow with glow */}
                    <div
                        className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-full"
                        style={{
                            width: 0,
                            height: 0,
                            borderLeft: '14px solid transparent',
                            borderRight: '14px solid transparent',
                            borderTop: '14px solid rgba(0, 0, 0, 0.95)',
                            filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.5))'
                        }}
                    />
                </div>
            </Html>
        </group>
    )
}

// Main 3D Scene Component
const Scene3D: React.FC<{
    points: EmbeddingPoint[]
    selectedPoint: EmbeddingPoint | null
    hoveredPoint: EmbeddingPoint | null
    onPointClick: (point: EmbeddingPoint) => void
    onPointHover: (point: EmbeddingPoint | null) => void
    colorScheme: string
    nodeSize: number
    showLabels: boolean
    showConnections: boolean
    showAllConnections: boolean
    similarityThreshold: number
    animationSpeed: number
}> = ({
    points,
    selectedPoint,
    hoveredPoint,
    onPointClick,
    onPointHover,
    colorScheme,
    nodeSize,
    showLabels,
    showConnections,
    showAllConnections,
    similarityThreshold,
    animationSpeed,
}) => {
        const { camera } = useThree()
        const groupRef = useRef<THREE.Group>(null)
        const [hoveredPosition, setHoveredPosition] = useState<THREE.Vector3 | null>(null)

        // Auto-fit camera only once when points first load
        useEffect(() => {
            if (points.length > 0) {
                const box = new THREE.Box3()
                points.forEach(point => {
                    box.expandByPoint(new THREE.Vector3(...point.position_3d))
                })

                const center = box.getCenter(new THREE.Vector3())
                const size = box.getSize(new THREE.Vector3())
                const maxDim = Math.max(size.x, size.y, size.z)

                // Only set camera position once, don't continuously update
                if (camera.position.length() < 10) { // Only if camera hasn't been moved by user
                    camera.position.set(center.x + maxDim * 1.5, center.y + maxDim, center.z + maxDim * 1.5)
                    camera.lookAt(center)
                }
            }
        }, [points.length]) // Only depend on points.length, not the full points array

        useFrame((_, delta) => {
            if (!groupRef.current || animationSpeed <= 0 || selectedPoint) {
                return
            }

            groupRef.current.rotation.y += delta * 0.05 * animationSpeed
        })

        // Update hovered position for tooltip - stable version
        useEffect(() => {
            if (hoveredPoint) {
                const position = new THREE.Vector3(...hoveredPoint.position_3d)
                setHoveredPosition(position)
            } else {
                setHoveredPosition(null)
            }
        }, [hoveredPoint?.entry_id]) // Only update when the hovered point ID changes

        const highlightedPoints = useMemo(() => {
            if (!selectedPoint) return new Set<string>()

            const highlighted = new Set<string>()
            points.forEach(point => {
                if (point.entry_id !== selectedPoint.entry_id) {
                    let similarity = 0
                    if (point.category === selectedPoint.category) similarity += 0.5
                    const commonTags = point.tags.filter(tag => selectedPoint.tags.includes(tag)).length
                    if (point.tags.length > 0 && selectedPoint.tags.length > 0) {
                        similarity += (commonTags / Math.max(point.tags.length, selectedPoint.tags.length)) * 0.5
                    }

                    if (similarity >= similarityThreshold) {
                        highlighted.add(point.entry_id)
                    }
                }
            })

            return highlighted
        }, [selectedPoint?.entry_id, points.length, similarityThreshold]) // Only recalculate when necessary

        return (
            <>
                {/* Local-only lighting and atmosphere (no external HDR dependency). */}
                <ambientLight intensity={0.28} />
                <hemisphereLight args={['#bfdbfe', '#0b1220', 0.35]} />
                <pointLight position={[20, 24, 18]} intensity={0.55} color="#f8fafc" />
                <pointLight position={[-24, -16, -18]} intensity={0.24} color="#38bdf8" />
                <Sparkles
                    count={120}
                    scale={[220, 140, 220]}
                    size={2.1}
                    speed={0.14}
                    opacity={0.26}
                    color="#93c5fd"
                />

                {/* Main group with all nodes */}
                <group ref={groupRef}>
                    {points.map((point) => (
                        <EmbeddingNode
                            key={point.entry_id}
                            point={point}
                            isSelected={selectedPoint?.entry_id === point.entry_id}
                            isHighlighted={highlightedPoints.has(point.entry_id)}
                            onClick={onPointClick}
                            onHover={onPointHover}
                            colorScheme={colorScheme}
                            nodeSize={nodeSize}
                            showLabels={showLabels}
                        />
                    ))}

                    {/* Connection lines */}
                    {showConnections && (
                        <ConnectionLines
                            points={points}
                            selectedPoint={selectedPoint}
                            similarityThreshold={similarityThreshold}
                            showAllConnections={showAllConnections}
                        />
                    )}
                </group>

                {/* Hover tooltip */}
                <HoverTooltip point={hoveredPoint} position={hoveredPosition} />
            </>
        )
    }

// Main Component
interface Advanced3DVisualizationProps {
    isOpen: boolean
    onClose: () => void
}

export const Advanced3DVisualization: React.FC<Advanced3DVisualizationProps> = ({
    isOpen,
    onClose
}) => {
    const [points, setPoints] = useState<EmbeddingPoint[]>([])
    const [processedPoints, setProcessedPoints] = useState<EmbeddingPoint[]>([])
    const [selectedPoint, setSelectedPoint] = useState<EmbeddingPoint | null>(null)
    const [hoveredPoint, setHoveredPoint] = useState<EmbeddingPoint | null>(null)
    const [selectedDetails, setSelectedDetails] = useState<EmbeddingDetails | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const [categoryFilter, setCategoryFilter] = useState<string>('all')
    const [typeFilter, setTypeFilter] = useState<string>('all')
    const [colorScheme, setColorScheme] = useState<string>('category')
    const [showLabels, setShowLabels] = useState(false)
    const [showConnections, setShowConnections] = useState(true)
    const [showAllConnections, setShowAllConnections] = useState(false)
    const [nodeSize, setNodeSize] = useState(1.15)
    const [similarityThreshold, setSimilarityThreshold] = useState(0.3)
    const [animationSpeed, setAnimationSpeed] = useState(1)
    const [isFullscreen, setIsFullscreen] = useState(false)
    const [showStats, setShowStats] = useState(false)
    const [categories, setCategories] = useState<string[]>([])
    const [types, setTypes] = useState<string[]>([])
    const [viewMode, setViewMode] = useState<'graph' | 'list'>('graph')
    const [isControlPanelOpen, setIsControlPanelOpen] = useState(true)
    const [isMobileViewport, setIsMobileViewport] = useState(false)
    const hoverStabilityRef = useRef<NodeJS.Timeout | null>(null)

    const clearFocusedSelection = () => {
        setSelectedPoint(null)
        setSelectedDetails(null)
        setHoveredPoint(null)
    }

    // Stable hover handler to prevent flickering
    const handlePointHover = (point: EmbeddingPoint | null) => {
        if (selectedPoint) {
            if (point && point.entry_id === selectedPoint.entry_id) {
                setHoveredPoint(point)
            }
            return
        }

        if (hoverStabilityRef.current) {
            clearTimeout(hoverStabilityRef.current)
        }

        if (point) {
            // Immediately show hover
            setHoveredPoint(point)
        } else {
            // Delay hiding to prevent flickering
            hoverStabilityRef.current = setTimeout(() => {
                setHoveredPoint(null)
            }, 150)
        }
    }

    // Load embeddings data when component opens
    useEffect(() => {
        if (isOpen) {
            loadEmbeddingsData()
            if (typeof window !== 'undefined' && window.innerWidth < 1024) {
                setIsControlPanelOpen(false)
            }
        }
    }, [isOpen])

    useEffect(() => {
        if (typeof window === 'undefined') {
            return
        }

        const syncViewportState = () => {
            const compact = window.innerWidth < 1024
            setIsMobileViewport(compact)

            if (!compact) {
                setIsControlPanelOpen(true)
            }
        }

        syncViewportState()
        window.addEventListener('resize', syncViewportState)
        return () => {
            window.removeEventListener('resize', syncViewportState)
        }
    }, [])

    // Cleanup hover timeout on unmount
    useEffect(() => {
        return () => {
            if (hoverStabilityRef.current) {
                clearTimeout(hoverStabilityRef.current)
            }
        }
    }, [])

    // Preserve semantic coordinates when available, otherwise build deterministic fallback coordinates.
    const processPointsForVisualization = (rawPoints: EmbeddingPoint[]): EmbeddingPoint[] => {
        if (rawPoints.length === 0) {
            return []
        }

        if (hasMeaningfulCoordinates(rawPoints)) {
            return normalizeSemanticPositions(rawPoints)
        }

        return buildStableFallbackLayout(rawPoints)
    }

    // Process points when raw points change
    useEffect(() => {
        if (points.length > 0) {
            const processed = processPointsForVisualization(points)
            setProcessedPoints(processed)
        }
    }, [points])

    // Filter points based on search and filters
    const filteredPoints = processedPoints.filter(point => {
        const matchesSearch = searchQuery === '' ||
            point.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            point.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
            point.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))

        const matchesCategory = categoryFilter === 'all' || point.category === categoryFilter
        const matchesType = typeFilter === 'all' || point.entry_type === typeFilter

        return matchesSearch && matchesCategory && matchesType
    })

    const visualizationHealth = useMemo(() => {
        const hasSemanticCoordinates = hasMeaningfulCoordinates(points)
        const nonZeroEmbeddingCount = points.filter((point) => hasNonZeroEmbedding(point)).length

        return {
            hasSemanticCoordinates,
            usesFallbackLayout: points.length > 0 && !hasSemanticCoordinates,
            nonZeroEmbeddingCount,
            embeddingCoverage: points.length > 0 ? nonZeroEmbeddingCount / points.length : 0,
        }
    }, [points])

    const topCategoryCounts = useMemo(() => {
        const counts = filteredPoints.reduce<Record<string, number>>((acc, point) => {
            acc[point.category] = (acc[point.category] || 0) + 1
            return acc
        }, {})

        return Object.entries(counts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
    }, [filteredPoints])

    const applyEmbeddingsPayload = (data: EmbeddingPoint[]) => {
        const normalizedData = data.map((point) => {
            const normalizedCategory = normalizePointCategory(point.category, point.entry_type, point.tags)
            const normalizedType = normalizePointType(point.entry_type, normalizedCategory, point.tags)

            return {
                ...point,
                category: normalizedCategory,
                entry_type: normalizedType,
            }
        })

        setPoints(normalizedData)
        const uniqueCategories = [...new Set(normalizedData.map((point) => point.category))] as string[]
        const uniqueTypes = [...new Set(normalizedData.map((point) => point.entry_type))] as string[]
        setCategories(uniqueCategories)
        setTypes(uniqueTypes)
    }

    const loadEmbeddingsData = async () => {
        setIsLoading(true)
        const cachedData = readEmbeddingsCache()
        const hasCachedData = Array.isArray(cachedData) && cachedData.length > 0

        if (hasCachedData) {
            applyEmbeddingsPayload(cachedData as EmbeddingPoint[])
            setIsLoading(false)
        }

        try {
            const response = await fetch('/api/knowledge/embeddings/visualization')
            if (response.ok) {
                const data = (await response.json()) as EmbeddingPoint[]
                applyEmbeddingsPayload(data)
                writeEmbeddingsCache(data)

                if (data.length === 0) {
                    toast.info('No embeddings available yet. Add some knowledge entries first!')
                } else if (!hasCachedData) {
                    toast.success(`Loaded ${data.length} embeddings for 3D visualization`)
                }
            } else {
                throw new Error('Failed to load embeddings data')
            }
        } catch (error) {
            console.error('Failed to load embeddings:', error)
            if (!hasCachedData) {
                toast.error('Failed to load embeddings data')
            }
        } finally {
            if (!hasCachedData) {
                setIsLoading(false)
            }
        }
    }

    const handlePointClick = async (point: EmbeddingPoint) => {
        setSelectedPoint(point)
        setHoveredPoint(point)

        try {
            const response = await fetch(`/api/knowledge/embeddings/${point.entry_id}/details`)
            if (response.ok) {
                const details = await response.json()
                const normalizedCategory = normalizePointCategory(
                    details?.entry?.category,
                    details?.entry?.entry_type,
                    details?.entry?.tags || [],
                )
                const normalizedType = normalizePointType(
                    details?.entry?.entry_type,
                    normalizedCategory,
                    details?.entry?.tags || [],
                )

                const normalizedSimilarEntries = Array.isArray(details?.similar_entries)
                    ? details.similar_entries.map((entry: any) => {
                        const similarCategory = normalizePointCategory(
                            entry?.category,
                            entry?.entry_type,
                            [],
                        )
                        return {
                            ...entry,
                            category: similarCategory,
                            entry_type: normalizePointType(entry?.entry_type, similarCategory, []),
                        }
                    })
                    : []

                setSelectedDetails({
                    ...details,
                    entry: {
                        ...details.entry,
                        category: normalizedCategory,
                        entry_type: normalizedType,
                    },
                    similar_entries: normalizedSimilarEntries,
                })
            }
        } catch (error) {
            console.error('Failed to load point details:', error)
            toast.error('Failed to load embedding details')
        }
    }

    const resetView = () => {
        clearFocusedSelection()
        setSearchQuery('')
        setCategoryFilter('all')
        setTypeFilter('all')
    }

    const toggleFullscreen = () => {
        setIsFullscreen(!isFullscreen)
    }

    if (!isOpen) return null

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={`fixed inset-0 bg-black/95 flex z-50 relative ${isFullscreen ? 'p-0' : 'p-2 sm:p-4'}`}
        >
            {/* Main View Area */}
            <div className="flex-1 relative">
                {isLoading ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center text-white">
                            <div className="animate-spin w-12 h-12 border-2 border-white border-t-transparent rounded-full mx-auto mb-4" />
                            <p>Loading visualization...</p>
                        </div>
                    </div>
                ) : filteredPoints.length === 0 ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center text-white">
                            <div className="w-16 h-16 mx-auto mb-4 opacity-50">
                                <Settings className="w-full h-full" />
                            </div>
                            <h3 className="text-xl font-semibold mb-2">No Embeddings Available</h3>
                            <p className="text-gray-300 mb-4 max-w-md">
                                The knowledge base doesn't have any embeddings yet.
                                Add some knowledge entries and they will appear here.
                            </p>
                        </div>
                    </div>
                ) : viewMode === 'graph' ? (
                    <Canvas
                        camera={{ position: [80, 50, 80], fov: 60 }}
                        gl={{ antialias: true, alpha: false }}
                        onPointerMissed={(event) => {
                            if (event.type === 'click') {
                                clearFocusedSelection()
                            }
                        }}
                        onCreated={({ gl, scene }) => {
                            gl.setClearColor('#0a0a0a')
                            scene.fog = new THREE.Fog('#0a0a0a', 100, 300)
                        }}
                    >
                        <Suspense fallback={null}>
                            <Scene3D
                                points={filteredPoints}
                                selectedPoint={selectedPoint}
                                hoveredPoint={hoveredPoint}
                                onPointClick={handlePointClick}
                                onPointHover={handlePointHover}
                                colorScheme={colorScheme}
                                nodeSize={nodeSize}
                                showLabels={showLabels}
                                showConnections={showConnections}
                                showAllConnections={showAllConnections}
                                similarityThreshold={similarityThreshold}
                                animationSpeed={animationSpeed}
                            />
                            <OrbitControls
                                enablePan
                                enableZoom
                                enableRotate
                                maxDistance={300}
                                minDistance={20}
                                dampingFactor={0.05}
                                enableDamping
                            />
                            {showStats && <Stats />}
                        </Suspense>
                    </Canvas>
                ) : (
                    // List View
                    <div className="flex h-full flex-col bg-gray-900 lg:flex-row">
                        {/* Left Panel - List of Embeddings */}
                        <div className="w-full border-b border-gray-700 overflow-y-auto lg:w-1/2 lg:border-b-0 lg:border-r">
                            <div className="p-4">
                                <h3 className="text-white font-semibold mb-4">Knowledge Embeddings ({filteredPoints.length})</h3>
                                <div className="space-y-2">
                                    {filteredPoints.map((point) => (
                                        <div
                                            key={point.entry_id}
                                            onClick={() => handlePointClick(point)}
                                            className={`p-3 rounded-lg cursor-pointer transition-colors ${selectedPoint?.entry_id === point.entry_id
                                                ? 'bg-blue-600/30 border border-blue-500'
                                                : 'bg-gray-800 hover:bg-gray-700'
                                                }`}
                                        >
                                            <div className="flex items-start justify-between">
                                                <div className="flex-1">
                                                    <h4 className="text-white font-medium text-sm mb-1">{point.title}</h4>
                                                    <p className="text-gray-300 text-xs mb-2 line-clamp-2">
                                                        {point.content.length > 100
                                                            ? point.content.substring(0, 100) + '...'
                                                            : point.content
                                                        }
                                                    </p>
                                                    <div className="flex flex-wrap gap-1">
                                                        <Badge variant="outline" className="text-xs bg-blue-500/20 text-blue-300 border-blue-500/50">
                                                            {toDisplayLabel(point.category)}
                                                        </Badge>
                                                        <Badge variant="outline" className="text-xs bg-amber-500/20 text-amber-300 border-amber-500/50">
                                                            {toDisplayLabel(point.entry_type)}
                                                        </Badge>
                                                    </div>
                                                </div>
                                                <div className="ml-2">
                                                    <div
                                                        className="w-3 h-3 rounded-full"
                                                        style={{
                                                            backgroundColor: getCategoryColor(point.category)
                                                        }}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Right Panel - Details */}
                        <div className="w-full overflow-y-auto lg:w-1/2">
                            <div className="p-4">
                                {selectedPoint && selectedDetails ? (
                                    <div className="text-white">
                                        <h3 className="font-semibold mb-4">Embedding Details</h3>

                                        <div className="space-y-4">
                                            <div>
                                                <h4 className="font-medium text-lg mb-2">{selectedDetails.entry.title}</h4>
                                                <p className="text-gray-300 text-sm leading-relaxed">
                                                    {selectedDetails.entry.content}
                                                </p>
                                            </div>

                                            <div>
                                                <h5 className="font-medium mb-2">Metadata</h5>
                                                <div className="grid grid-cols-2 gap-2 text-xs">
                                                    <div>
                                                        <span className="text-gray-400">Category:</span>
                                                        <span className="ml-2 text-blue-300">{toDisplayLabel(selectedDetails.entry.category)}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-400">Type:</span>
                                                        <span className="ml-2 text-amber-300">{toDisplayLabel(selectedDetails.entry.entry_type)}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-400">Content Length:</span>
                                                        <span className="ml-2">{selectedDetails.statistics.content_length}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-400">Tags:</span>
                                                        <span className="ml-2">{selectedDetails.statistics.tag_count}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-400">Embedding Dim:</span>
                                                        <span className="ml-2">{selectedDetails.embedding_info.embedding_dimension}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-400">Created:</span>
                                                        <span className="ml-2">{new Date(selectedDetails.entry.created_at).toLocaleDateString()}</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <div>
                                                <h5 className="font-medium mb-2">Tags</h5>
                                                <div className="flex flex-wrap gap-1">
                                                    {selectedDetails.entry.tags.map(tag => (
                                                        <Badge key={tag} variant="outline" className="text-xs bg-gray-700 text-gray-300">
                                                            {tag}
                                                        </Badge>
                                                    ))}
                                                </div>
                                            </div>

                                            {selectedDetails.similar_entries.length > 0 && (
                                                <div>
                                                    <h5 className="font-medium mb-2">Similar Entries</h5>
                                                    <div className="space-y-2">
                                                        {selectedDetails.similar_entries.map(similar => (
                                                            <div key={similar.entry_id} className="p-2 bg-gray-800 rounded text-xs">
                                                                <div className="font-medium">{similar.title}</div>
                                                                <div className="text-gray-400">
                                                                    Similarity: {(similar.similarity_score * 100).toFixed(1)}% • {toDisplayLabel(similar.category)}
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-center text-gray-400 mt-8">
                                        <Box className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                        <p>Select an embedding from the list to view details</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* Control Overlay */}
                <div className="absolute top-4 left-4 flex flex-col gap-2">
                    <Button
                        variant="secondary"
                        size="icon"
                        onClick={onClose}
                        className="bg-black/50 hover:bg-black/70 text-white"
                    >
                        <X className="w-5 h-5" />
                    </Button>

                    <Button
                        variant="secondary"
                        size="icon"
                        onClick={resetView}
                        className="bg-black/50 hover:bg-black/70 text-white"
                    >
                        <RotateCcw className="w-5 h-5" />
                    </Button>

                    <Button
                        variant="secondary"
                        size="icon"
                        onClick={toggleFullscreen}
                        className="bg-black/50 hover:bg-black/70 text-white"
                    >
                        {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
                    </Button>

                    <Button
                        variant="secondary"
                        size="icon"
                        onClick={() => setIsControlPanelOpen((prev) => !prev)}
                        className="bg-black/50 hover:bg-black/70 text-white"
                        title={isControlPanelOpen ? 'Hide controls panel' : 'Show controls panel'}
                    >
                        {isControlPanelOpen ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
                    </Button>
                </div>

                {/* Graph/List Toggle - Top Right */}
                <div className="absolute top-2 right-2 flex gap-2 mb-4 sm:top-4 sm:right-4">
                    <div className="bg-black/80 rounded-lg p-1 border border-gray-700">
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setViewMode('graph')}
                            className={viewMode === 'graph' ? 'text-white bg-blue-600 hover:bg-blue-700' : 'text-gray-400 hover:text-white hover:bg-gray-700'}
                        >
                            Graph
                        </Button>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setViewMode('list')}
                            className={viewMode === 'list' ? 'text-white bg-blue-600 hover:bg-blue-700' : 'text-gray-400 hover:text-white hover:bg-gray-700'}
                        >
                            List
                        </Button>
                    </div>
                </div>

                {points.length > 0 && (
                    <div className="absolute left-1/2 top-3 z-20 w-[min(40rem,calc(100vw-2rem))] -translate-x-1/2 px-2 sm:top-4 sm:px-0">
                        {visualizationHealth.usesFallbackLayout ? (
                            <div className="rounded-lg border border-amber-400/40 bg-amber-950/70 px-3 py-2 text-xs text-amber-100 backdrop-blur-sm sm:text-sm">
                                <div className="flex items-center gap-2 font-medium">
                                    <AlertCircle className="h-4 w-4" />
                                    Fallback layout active
                                </div>
                                <p className="mt-1 text-amber-100/90">
                                    Semantic coordinates are not distinct yet, so the graph is currently clustered by category.
                                </p>
                            </div>
                        ) : (
                            <div className="rounded-lg border border-emerald-400/40 bg-emerald-950/65 px-3 py-2 text-xs text-emerald-100 backdrop-blur-sm sm:text-sm">
                                <div className="flex items-center gap-2 font-medium">
                                    <CheckCircle2 className="h-4 w-4" />
                                    Semantic layout active
                                </div>
                                <p className="mt-1 text-emerald-100/90">
                                    Node positions are driven by backend embedding coordinates.
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {viewMode === 'graph' && selectedPoint && selectedDetails && (
                    <div className="absolute bottom-3 left-3 z-20 w-[min(30rem,calc(100vw-1.5rem))] rounded-2xl border border-cyan-300/40 bg-slate-950/85 p-3 text-slate-100 shadow-2xl backdrop-blur-sm sm:bottom-4 sm:left-4 sm:p-4">
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <p className="text-[11px] uppercase tracking-[0.16em] text-cyan-300">Focused Node</p>
                                <h4 className="mt-1 text-lg font-semibold leading-snug">{selectedDetails.entry.title}</h4>
                            </div>
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 px-2 text-slate-200 hover:bg-white/10 hover:text-white"
                                onClick={clearFocusedSelection}
                            >
                                Clear
                            </Button>
                        </div>

                        <p className="mt-2 text-sm leading-relaxed text-slate-200">
                            {selectedDetails.entry.content.length > 240
                                ? `${selectedDetails.entry.content.substring(0, 240)}...`
                                : selectedDetails.entry.content}
                        </p>

                        <div className="mt-3 flex flex-wrap gap-2">
                            <Badge variant="outline" className="border-cyan-300/40 bg-cyan-950/40 text-cyan-100">
                                {toDisplayLabel(selectedDetails.entry.category)}
                            </Badge>
                            <Badge variant="outline" className="border-amber-300/40 bg-amber-950/35 text-amber-100">
                                {toDisplayLabel(selectedDetails.entry.entry_type)}
                            </Badge>
                            <Badge variant="outline" className="border-slate-300/35 bg-slate-900/50 text-slate-200">
                                {selectedDetails.statistics.tag_count} tags
                            </Badge>
                        </div>

                        <p className="mt-3 text-xs text-slate-300">
                            Focus stays locked until you click empty space in the graph or press clear.
                        </p>
                    </div>
                )}

                {/* Legend Panel - Only show in graph mode */}
                {viewMode === 'graph' && (
                    <div className="absolute right-2 top-16 w-[min(18rem,calc(100vw-1rem))] text-white text-sm bg-black/80 rounded-lg p-3 sm:p-4 sm:right-4 sm:w-auto sm:min-w-[200px] border border-gray-700">
                        <div className="flex items-center gap-2 mb-3">
                            <Info className="w-4 h-4" />
                            <span className="font-semibold">Legend</span>
                        </div>

                        {/* Statistics */}
                        <div className="mb-4">
                            <div className="text-xs text-gray-400 mb-2">STATISTICS</div>
                            <div className="text-xs space-y-1">
                                <div className="flex justify-between">
                                    <span className="text-blue-300">{filteredPoints.length} visible nodes</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-emerald-300">{topCategoryCounts.length} active categories</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-cyan-300">{showConnections ? 'Connections on' : 'Connections off'}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-amber-300">
                                        {visualizationHealth.usesFallbackLayout ? 'Category fallback layout' : 'Semantic layout'}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-sky-300">
                                        Embedding coverage {(visualizationHealth.embeddingCoverage * 100).toFixed(0)}%
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Node Types */}
                        <div className="mb-4">
                            <div className="text-xs text-gray-400 mb-2">NODES</div>
                            <div className="text-xs space-y-1">
                                {topCategoryCounts.map(([category, count]) => (
                                    <div key={category} className="flex items-center justify-between gap-2">
                                        <div className="flex items-center gap-2">
                                            <div
                                                className="w-2 h-2 rounded-full"
                                                style={{ backgroundColor: getCategoryColor(category) }}
                                            ></div>
                                            <span className="capitalize">{toDisplayLabel(category)}</span>
                                        </div>
                                        <span className="text-gray-400">{count}</span>
                                    </div>
                                ))}
                                {topCategoryCounts.length === 0 && (
                                    <div className="text-gray-400">No categories in current filter</div>
                                )}
                                {topCategoryCounts.length > 0 && filteredPoints.length > 0 && (
                                    <div className="pt-1 text-[11px] text-gray-400">
                                        Top categories by visible nodes
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Status */}
                        <div className="mb-4">
                            <div className="text-xs text-gray-400 mb-2">STATUS</div>
                            <div className="text-xs space-y-1">
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-red-400"></div>
                                    <span>Selected</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-green-400"></div>
                                    <span>Similar</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-gray-400"></div>
                                    <span>Default</span>
                                </div>
                            </div>
                        </div>

                        {/* Connections */}
                        <div>
                            <div className="text-xs text-gray-400 mb-2">CONNECTIONS</div>
                            <div className="text-xs space-y-1">
                                <div className="flex items-center gap-2">
                                    <div className="w-4 h-0.5 bg-green-400"></div>
                                    <span>Strong</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-4 h-0.5 bg-blue-400"></div>
                                    <span>Weak</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {isMobileViewport && isControlPanelOpen && (
                <button
                    type="button"
                    aria-label="Close controls panel"
                    className="absolute inset-0 z-30 bg-black/40 lg:hidden"
                    onClick={() => setIsControlPanelOpen(false)}
                />
            )}

            {/* Control Panel */}
            <div
                className={`bg-background border-l border-border overflow-y-auto transition-all duration-300 ${isMobileViewport
                        ? `${isFullscreen ? 'w-80' : 'w-96'} max-w-[92vw] fixed right-0 top-0 z-40 h-full shadow-2xl`
                        : `${isControlPanelOpen ? (isFullscreen ? 'w-80' : 'w-96') : 'w-0'} h-full`
                    } ${isMobileViewport && !isControlPanelOpen ? 'pointer-events-none translate-x-full' : 'translate-x-0'} ${!isControlPanelOpen && !isMobileViewport ? 'border-l-0' : ''}`}
            >
                <div className={`p-4 space-y-4 ${!isControlPanelOpen && !isMobileViewport ? 'hidden' : ''}`}>
                    {/* Search and Filters */}
                    <Card className="p-4">
                        <h3 className="font-semibold mb-3">Filters & Search</h3>

                        <div className="space-y-3">
                            <div>
                                <label className="block text-sm font-medium mb-1">Search</label>
                                <div className="relative">
                                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                    <Input
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        placeholder="Search embeddings..."
                                        className="pl-10"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium mb-1">Category</label>
                                <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="all">All Categories</SelectItem>
                                        {categories.map(category => (
                                            <SelectItem key={category} value={category}>
                                                {toDisplayLabel(category)}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium mb-1">Type</label>
                                <Select value={typeFilter} onValueChange={setTypeFilter}>
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="all">All Types</SelectItem>
                                        {types.map(type => (
                                            <SelectItem key={type} value={type}>
                                                {toDisplayLabel(type)}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>
                    </Card>

                    {/* Visualization Settings */}
                    <Card className="p-4">
                        <h3 className="font-semibold mb-3">Visualization</h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium mb-1">Color Scheme</label>
                                <Select value={colorScheme} onValueChange={setColorScheme}>
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="category">By Category</SelectItem>
                                        <SelectItem value="type">By Type</SelectItem>
                                        <SelectItem value="age">By Age</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium mb-2">Node Size</label>
                                <Slider
                                    value={[nodeSize]}
                                    onValueChange={(value) => setNodeSize(value[0])}
                                    min={0.5}
                                    max={2}
                                    step={0.1}
                                    className="w-full"
                                />
                                <div className="text-xs text-muted-foreground mt-1">{nodeSize.toFixed(1)}</div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium mb-2">Similarity Threshold</label>
                                <Slider
                                    value={[similarityThreshold]}
                                    onValueChange={(value) => setSimilarityThreshold(value[0])}
                                    min={0.1}
                                    max={0.9}
                                    step={0.1}
                                    className="w-full"
                                />
                                <div className="text-xs text-muted-foreground mt-1">{(similarityThreshold * 100).toFixed(0)}%</div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium mb-2">Animation Speed</label>
                                <Slider
                                    value={[animationSpeed]}
                                    onValueChange={(value) => setAnimationSpeed(value[0])}
                                    min={0}
                                    max={3}
                                    step={0.1}
                                    className="w-full"
                                />
                                <div className="text-xs text-muted-foreground mt-1">
                                    {animationSpeed === 0 ? 'Paused' : `${animationSpeed.toFixed(1)}x`}
                                </div>
                            </div>

                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <label className="text-sm font-medium">Show Labels</label>
                                    <Switch
                                        checked={showLabels}
                                        onCheckedChange={setShowLabels}
                                    />
                                </div>

                                <div className="flex items-center justify-between">
                                    <label className="text-sm font-medium">Show Connections</label>
                                    <Switch
                                        checked={showConnections}
                                        onCheckedChange={setShowConnections}
                                    />
                                </div>

                                <div className="flex items-center justify-between">
                                    <label className="text-sm font-medium">All Connections</label>
                                    <Switch
                                        checked={showAllConnections}
                                        onCheckedChange={setShowAllConnections}
                                        disabled={!showConnections}
                                    />
                                </div>

                                <div className="flex items-center justify-between">
                                    <label className="text-sm font-medium">Performance Stats</label>
                                    <Switch
                                        checked={showStats}
                                        onCheckedChange={setShowStats}
                                    />
                                </div>
                            </div>
                        </div>
                    </Card>

                    {/* Selected Point Details */}
                    {selectedPoint && selectedDetails && (
                        <Card className="p-4 max-h-[44vh] overflow-y-auto">
                            <h3 className="font-semibold mb-3">Selected Node</h3>

                            <div className="space-y-3">
                                <div>
                                    <h4 className="font-medium text-sm">{selectedDetails.entry.title}</h4>
                                    <p className="text-xs text-muted-foreground mt-1">
                                        {selectedDetails.entry.content.length > 150
                                            ? selectedDetails.entry.content.substring(0, 150) + '...'
                                            : selectedDetails.entry.content
                                        }
                                    </p>
                                </div>

                                <div className="flex flex-wrap gap-1">
                                    <Badge variant="outline">{toDisplayLabel(selectedDetails.entry.category)}</Badge>
                                    <Badge variant="secondary">{toDisplayLabel(selectedDetails.entry.entry_type)}</Badge>
                                    {selectedDetails.entry.tags.slice(0, 3).map(tag => (
                                        <Badge key={tag} variant="outline" className="text-xs">{tag}</Badge>
                                    ))}
                                    {selectedDetails.entry.tags.length > 3 && (
                                        <Badge variant="outline" className="text-xs">+{selectedDetails.entry.tags.length - 3}</Badge>
                                    )}
                                </div>

                                <div className="text-xs space-y-1">
                                    <div>Embedding Dimension: {selectedDetails.embedding_info.embedding_dimension}</div>
                                    <div>Content Length: {selectedDetails.statistics.content_length}</div>
                                    <div>Tags: {selectedDetails.statistics.tag_count}</div>
                                    <div>Created: {new Date(selectedDetails.entry.created_at).toLocaleDateString()}</div>
                                </div>

                                {selectedDetails.similar_entries.length > 0 && (
                                    <div>
                                        <h5 className="font-medium text-sm mb-2">Similar Entries</h5>
                                        <div className="space-y-1">
                                            {selectedDetails.similar_entries.map(similar => (
                                                <div key={similar.entry_id} className="text-xs p-2 bg-muted rounded border">
                                                    <div className="font-medium">{similar.title}</div>
                                                    <div className="text-muted-foreground">
                                                        Similarity: {(similar.similarity_score * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </Card>
                    )}

                    {/* Instructions */}
                    <Card className="p-4">
                        <h3 className="font-semibold mb-3">Controls</h3>
                        <div className="text-xs space-y-2 text-muted-foreground">
                            <div>• <strong>Mouse:</strong> Rotate view</div>
                            <div>• <strong>Scroll:</strong> Zoom in/out</div>
                            <div>• <strong>Right-click + drag:</strong> Pan</div>
                            <div>• <strong>Click point:</strong> Select and show details</div>
                            <div>• <strong>Hover point:</strong> Quick info</div>
                            <div>• <strong>Settings:</strong> Customize visualization</div>
                        </div>
                    </Card>
                </div>
            </div>
        </motion.div>
    )
}