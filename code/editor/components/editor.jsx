'use client'

import { loadSegmentation } from "@/lib/tiles";
import React, { useEffect, useState } from "react";
import { Layer, Rect, Stage, Text } from "react-konva"

const TILE_SIZE = 30
const GRID_ROWS = 14
const GRID_COLUMNS = 14
const GRID_WIDTH = GRID_COLUMNS * TILE_SIZE

const initGrid = () => {
  return Array.from({ length: GRID_ROWS }, (_, row) =>
    Array.from({ length: GRID_COLUMNS }, (_, col) => ({
      x: col * TILE_SIZE,
      y: row * TILE_SIZE,
      label: 0,
    }))
  )
}

const getColor = index => {
  if (!index) {
    return 'white'
  }

  const label = labels[Number(index)]
  if(label?.color) {
    console.log(label)
    return label.color
  }

  const gray_scale = index / (17 - 1) * 255

  return `rgb(${gray_scale},${gray_scale},${gray_scale})`
}

const getTextColor = index => {
  if (!index) {
    return 'black'
  }

  const gray_scale = index / (17 - 1) * 255

  if (gray_scale > 125) {
    return 'black'
  }

  return 'white'
}

const labels = [
  {
    id: 0,
    displayName: 'None',
    color: '#FFF'
  },
  {
    id: 1,
    displayName: 'Industrie- und Gewerbeareal',
    color: '#C9C9C9'
  },
  {
    id: 2,
    displayName: 'Gebäudeareal',
    color: '#949494'
  },
  {
    id: 3,
    displayName: 'Verkehrsflächen',
    color: '#535353'
  },
  {
    id: 4,
    displayName: 'Besondere Siedlungsflächen',
    color: '#A04A00'
  },
  {
    id: 5,
    displayName: 'Erholungs- und Grünanlagen',
    color: '#21B600'
  },
  {
    id: 6,
    displayName: 'Obst-, Reb- und Gartenbauflächen',
    color: '#858C67'
  },
  {
    id: 7,
    displayName: 'Ackerland',
    color: '#C8AF8F'
  },
  {
    id: 8,
    displayName: 'Naturwiesen, Heimweiden',
    color: '#65A93F'
  },
  {
    id: 9,
    displayName: 'Alpwirtschaftsflächen',
    color: '#79A761'
  },
  {
    id: 10,
    displayName: 'Wald (ohne Gebüschwald)',
    color: '#495642'
  },
  {
    id: 11,
    displayName: 'Gebüschwald',
    color: '#1F400D'
  },
  {
    id: 12,
    displayName: 'Gehölze',
    color: '#744F18'
  },
  {
    id: 13,
    displayName: 'Stehende Gewässer',
    color: '#397381'
  },
  {
    id: 14,
    displayName: 'Fliessgewässer',
    color: '#286E7F'
  },
  {
    id: 15,
    displayName: 'Unproduktive Vegetation',
    color: '#1F400D'
  },
  {
    id: 16,
    displayName: 'Vegetationslose Flächen',
    color: '#744F18'
  },
  {
    id: 17,
    displayName: 'Gletscher, Firn',
    color: '#BEBFBF'
  }
]

const loadNewSegmentation = async (position, grid, setGrid) => {
  if (!position) return
  const segmentation = await loadSegmentation(position)
  const newGrid = [...grid]

  for (let row = 0; row < segmentation.length; row++) {
    for (let col = 0; col < segmentation[row].length; col++) {
      newGrid[row][col] = {
        ...newGrid[row][col],
        label: segmentation[row][col]
      }
    }
  }

  setGrid([...newGrid])
}


export const Editor = ({ position, setSegmentation, isLoading }) => {
  const [grid, setGrid] = useState(initGrid, [])
  const [currentLabel, setCurrentLabel] = useState(1)
  const [isDragging, setIsDragging] = useState(false)

  const handleMouseDown = () => setIsDragging(true)
  const handleMouseUp = () => setIsDragging(false)

  useEffect(() => {
    loadNewSegmentation(position, grid, setGrid)
  }, [position])


  useEffect(() => {
    setSegmentation(grid.map(row => row.map(cell => Number(cell.label))))
  }, [grid])

  const reset = () => {
    loadNewSegmentation(position, grid, setGrid)
  }


  const handleMouseMove = (event) => {
    if (!isDragging) return;

    const { x, y } = event.target.getStage().getPointerPosition();

    const col = Math.floor(x / TILE_SIZE);
    const row = Math.floor(y / TILE_SIZE);

    if (row >= 0 && row < GRID_ROWS && col >= 0 && col < GRID_COLUMNS) {
      setGrid((prevGrid) => {
        const newGrid = [...prevGrid];
        if (newGrid[row][col].label !== currentLabel) {
          newGrid[row][col] = { ...newGrid[row][col], label: currentLabel };
        }
        return newGrid;
      });
    }
  };

  return (
    <div className="px-4 py-4 flex flex-col gap-4 w-full flex-1">
      <h1 className="text-2xl font-light">Editor</h1>

      <Stage
        width={GRID_WIDTH}
        height={GRID_WIDTH}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
      >

        <Layer fill="black">
          {grid.flat().map((tile, index) => (
            <React.Fragment key={index}>
              <Rect
                x={tile.x}
                y={tile.y}
                width={TILE_SIZE}
                height={TILE_SIZE}
                fill={getColor(tile.label)}
                stroke="#bbb"
                strokeWidth={1}
              />
              <Text
                x={tile.x}
                y={tile.y}
                width={TILE_SIZE}
                height={TILE_SIZE}
                text={tile.label}
                fill={getTextColor(tile.label)}
                fontSize={12}
                align="center"
                verticalAlign="middle"
              />
            </React.Fragment>
          ))}

        </Layer>
      </Stage>

      <div>
        <label className="flex flex-col w-full">
          <span className="text-sm text-gray-800">
            Select Class
          </span>

          <select
            value={currentLabel}
            onChange={(e) => setCurrentLabel(e.target.value)}
            className="border p-2 rounded"
          >
            {labels.map((label) => (
              <option key={label.id} value={label.id}>
                {label.id} - {label.displayName}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="flex gap-2 justify-between">
        <label className="flex flex-col w-full">
          <span className="text-sm text-gray-800">
            Reset to Original Segmentation
          </span>

          <button className="border p-2 rounded disabled:bg-gray-100 disabled:text-gray-400" onClick={reset} disabled={isLoading}>
            Reset
          </button>
        </label>
      </div>
    </div>
  )
}