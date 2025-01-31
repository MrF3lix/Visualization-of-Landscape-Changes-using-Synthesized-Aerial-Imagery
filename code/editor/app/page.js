'use client'

import { Editor } from "@/components/editor";
import { Result } from "@/components/result";
import { Selector } from "@/components/selector";
import { generateImage, reGenerateImage } from "@/lib/tiles";
import { useState } from "react"
import { useDebounce } from 'react-use'

export default function Home() {
  const [imageBase64, setImageBase64] = useState()
  const [position, setPosition] = useState()
  const [segmentation, setSegmentation] = useState()
  const [isLoading, setIsLoading] = useState(false)

  useDebounce(async () => {
    setIsLoading(true)
    const response = await generateImage(position, segmentation)
    setImageBase64(response.image)
    setIsLoading(false)
  }, 1000, [segmentation]);

  const regenerate = async () => {
    setIsLoading(true)
    const response = await reGenerateImage(position, segmentation, imageBase64)
    setImageBase64(response.image)
    setIsLoading(false)
  }


  return (
    <div className="w-full max-w-[1920px] mx-auto">
      <div className="flex gap-4 justify-between min-h-[100vh]">
        <Selector setPosition={setPosition} />
        <Editor position={position} setSegmentation={setSegmentation} isLoading={isLoading} />
        <Result imageBase64={imageBase64} regenerate={regenerate} isLoading={isLoading} />
      </div>
    </div>
  );
}

