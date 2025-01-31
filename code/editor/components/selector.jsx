
'use client'

import { useEffect, useState } from 'react'
import AsyncSelect from 'react-select/async'
import { WGStoLV95, } from 'swiss-projection'

import { loadLocations } from '@/lib/location'

const DEFAULT_EAST = 2504
const DEFAULT_NORTH = 1150

const loadOptions = (
  inputValue,
  callback
) => {
  setTimeout(async () => {
    const results = await loadLocations(inputValue)
    callback(results);
  }, 100);
}

const getTilePosition = (location) => {
  const coords = location?.value?.properties?.coordinates
  if (!coords) {
    return
  }

  const chCoords = WGStoLV95(location.value);

  let east = chCoords.geometry.coordinates[0]
  let north = chCoords.geometry.coordinates[1]

  east = Number(east.toString().substring(0, 4))
  north = Number(north.toString().substring(0, 4))

  return `${east}_${north}`
}

export const Selector = ({ setPosition }) => {
  const [east, setEast] = useState(DEFAULT_EAST)
  const [north, setNorth] = useState(DEFAULT_NORTH)
  const [location, setLocation] = useState()
  const [tilePosition, setTilePosition] = useState()

  useEffect(() => {
    if (!location) return

    const pos = getTilePosition(location)

    setEast(pos?.split('_')[0] || DEFAULT_EAST)
    setNorth(pos?.split('_')[1] || DEFAULT_NORTH)

  }, [location])

  useEffect(() => {
    const pos = `${east}_${north}`

    setTilePosition(pos)
    setPosition(pos)
  }, [east, north])

  return (
    <div className="px-4 py-4 flex flex-col gap-4 w-full flex-1">
      <h1 className="text-2xl font-light">Selector</h1>

      <img
        width={420}
        src={`http://127.0.0.1:8001/base/${tilePosition}`}
      />

      <label className="flex flex-col w-full">
        <span className="text-sm text-gray-800">
          Select Location
        </span>

        <AsyncSelect
          className="w-full"
          cacheOptions
          defaultOptions
          isClearable
          loadOptions={loadOptions}
          onChange={setLocation}
          placeholder="Search for a Location"
        />
      </label>

      <div className="flex gap-2 justify-between">
        <label className="flex flex-col w-full">
          <span className="text-sm text-gray-800">
            East Coordinates
          </span>

          <input
            className="border p-2 rounded"
            value={east}
            type='number'
            onChange={e => setEast(e.target.value)}
          />
        </label>

        <label className="flex flex-col w-full">
          <span className="text-sm text-gray-800">
            North Coordinates
          </span>

          <input
            className="border p-2 rounded"
            value={north}
            type='number'
            onChange={e => setNorth(e.target.value)}
          />
        </label>
      </div>
    </div>
  )
}