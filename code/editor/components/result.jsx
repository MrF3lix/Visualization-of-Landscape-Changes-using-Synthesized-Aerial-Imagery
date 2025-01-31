
'use client'

export const Result = ({ imageBase64, regenerate, isLoading }) => {

  return (
    <div className="px-4 py-4 flex flex-col gap-4 w-full flex-1">
      <h1 className="text-2xl font-light">Result</h1>

      <div className="w-[420px] h-[420px] bg-gray-50 flex justify-center items-center">

        {imageBase64 ? (
          <img width={420} src={`data:image/png;base64,${imageBase64}`} alt="Generated Image" />
        ) : (
          <p>Generating Image...</p>
        )}

      </div>
      <div className="flex gap-2 justify-between">
        <label className="flex flex-col w-full">
          <span className="text-sm text-gray-800">
            Regenerate from Result
          </span>

          <button className="border p-2 rounded disabled:bg-gray-100 disabled:text-gray-400" onClick={regenerate} disabled={isLoading}>
            Regenerate
          </button>
        </label>
      </div>
    </div>
  )
}