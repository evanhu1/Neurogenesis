🔴 Bugs

1. useLayoutEffect for camera tracking blocks paint and triggers a cascading
   synchronous re-render The useLayoutEffect on line 94 fires on every snapshot
   change when an organism is focused. The call chain is: panToWorldPoint →
   updateViewport → setViewport(next). Since setViewport is called inside a
   useLayoutEffect, React processes the resulting re-render synchronously before
   the browser paints. So every incoming WebSocket tick, while tracking, causes:
   (1) the normal render from setSnapshot, (2) the useLayoutEffect fires and
   triggers setViewport, (3) a second synchronous render from the viewport state
   update, (4) browser finally paints. That's two full React reconciliation
   passes blocking paint, per tick. The canvas doesn't even benefit from this —
   it's rendered by the RAF loop reading viewportRef.current, which is already
   updated synchronously in step (2) regardless of useLayoutEffect vs useEffect.
   Switching to useEffect would let the browser paint immediately after the
   first render, and the RAF loop would pick up the updated viewport ref on its
   next frame (~16ms later at worst, visually imperceptible).
2. Viewport useState triggers wasted React re-renders on every pan/zoom In
   useWorldViewport, line 2018: const [, setViewport] =
   useState<WorldViewport>(...). The value is destructured away — nothing in
   WorldCanvas's JSX or any effect depends on it. The RAF loop reads
   viewportRef.current directly. The cursor class depends on
   isSpacePressed/isPanningWorld, which are separate state values. Yet every
   call to updateViewport (every wheel tick, every mouse-move during panning)
   calls setViewport(next), triggering a full React reconciliation of
   WorldCanvas. During that render, React re-evaluates the JSX, diffs it, and
   finds zero DOM changes. This is pure waste — easily 30-60 wasted
   reconciliations per second during zoom/pan interactions. The fix is to delete
   the setViewport(next) call in updateViewport (or remove the useState
   entirely).

🟡 Performance Issues

1. RAF loop renders unconditionally at 60fps (highest impact) The draw loop
   (line 154) calls renderWorld every frame regardless of whether anything has
   changed. When the simulation is paused and the user isn't interacting, this
   redraws the entire hex grid, all entities, and all walls 60 times per second
   for zero visual change. Fix: add a dirty flag (a ref). Set it to true when
   any input changes (snapshot, viewport, visibility toggles). In the RAF
   callback, skip the draw when the flag is false. This will cut idle CPU usage
   by ~95%.
2. Canvas draw calls are issued per-entity instead of batched (second highest
   impact) In drawVisibleGrid (line 2306), every single hex gets its own cycle
   of beginPath → traceHex → fill → stroke. Each fill() and stroke() call
   flushes the accumulated path to the GPU as a separate draw operation. For a
   50×50 visible grid, that's ~2,500 hexes × 2 flushes = 5,000 GPU operations
   per frame. Additionally, ctx.fillStyle = EARTH_COLOR, ctx.strokeStyle =
   '#8a94a8', and ctx.lineWidth = 0.4 are re-assigned to identical constant
   values on every iteration. The fix is to batch by fill style: call
   ctx.fillStyle once, ctx.beginPath() once, loop traceHex to accumulate all hex
   paths, then call fill() once and stroke() once. That collapses ~5,000 flushes
   to 2. The same pattern applies to the walls loop and the plants loop. The
   organism loop can't be fully batched (varying colors per species), but could
   be grouped by species.
3. traceHex recomputes cos/sin for 6 fixed angles, per hex, per frame There are
   exactly 6 angles and they never change. At 2,500 visible hexes, that's 15,000
   Math.cos + 15,000 Math.sin calls per frame. Precompute the 6 unit vertex
   offsets once at module scope: tsconst HEX_UNIT_VERTICES = Array.from({
   length: 6 }, (_, i) => { const angle = (Math.PI / 180) * (60 * i - 30);
   return [Math.cos(angle), Math.sin(angle)] as const; }); Then traceHex becomes
   multiply-and-add only.
4. Organism rendering computes facing direction from hexCenter + Math.hypot per
   organism For each organism, hexCenter is called twice — once for position,
   once for the facing neighbor (line 2425) — solely to derive the direction
   unit vector. But hexCenter is a linear function of (q, r), so neighbor -
   center is constant for a given facing direction, independent of (q, r). The 6
   normalized direction unit vectors can be precomputed at module scope: tsconst
   FACING_UNIT_VECTORS: Record<FacingDirection, [number, number]> = { East: [1,
   0], NorthEast: [0.5 * SQRT_3 / Math.hypot(0.5 * SQRT_3, 1.5), -1.5 /
   Math.hypot(0.5 * SQRT_3, 1.5)], // ... etc }; This eliminates a hexCenter
   call, a subtraction, and a Math.hypot per organism per frame.
5. computeStableVisibleSpeciesIds is O(history × species²) per tick This
   function (line 991) iterates every history point (up to 2048) and calls
   visibleSpeciesIds.includes() — a linear array scan — multiple times per
   point. It's invoked from a useMemo whose history dependency changes every
   tick. Converting visibleSpeciesIds to a Set for O(1) lookups is a minimal
   fix. The better fix is to maintain the visible set incrementally (update it
   with just the new data point each tick, rather than reprocessing the entire
   history).
6. computeBaseHexSize ignores worldWidth, requiring extreme zoom ranges for
   large worlds The _worldWidth parameter (line 2238) is intentionally unused —
   hex size is canvas-relative, and the viewport zoom compensates. This works
   for worlds up to ~60 hexes wide. Beyond that, fitWorldToCanvas clamps the
   auto-fit zoom to START_FIT_MIN_WORLD_ZOOM (0.65), so the world won't fit on
   screen at startup. Beyond ~250 hexes, even the manual minimum zoom of 0.2
   can't show the full world. If large worlds are a use case, hex size should
   factor in worldWidth.

🟢 Refactoring Opportunities

1. useSimulationSession — 520+ line god hook managing 15+ state values It
   handles session lifecycle, WebSocket management, batch runs, archived worlds,
   organism focus, speed control, step progress, and error state in a single
   function. This creates two concrete problems: (a) every returned callback
   captures every piece of state in its closure, making it impossible for React
   to avoid stale-closure bugs without aggressive useCallback deps, and (b) any
   state change triggers a re-render of the component using this hook, even if
   the component only cares about one slice. Decompose into:
   useSimulationConnection (WS + session restore), useOrganismFocus,
   useBatchRuns, useArchivedWorlds. Each owns its state and exposes a minimal
   interface.
2. ControlPanel takes 22 props Direct consequence of the god hook. The batch run
   section, archived worlds list, speed controls, and step controls should each
   be standalone components that subscribe to their own state, eliminating the
   prop drilling. As-is, every parent re-render forces a full reconciliation of
   the entire control panel including all its children.
3. Nested state setter in TickDelta handler obscures data flow
   setSpeciesPopulationHistory called inside setSnapshot's updater (line 1492)
   works correctly, but it's a non-obvious pattern that makes the data flow hard
   to trace. Extract the population history update to run alongside setSnapshot,
   using the computed nextSnapshot in a let variable.
4. normalizeSpeciesCounts is a no-op Line 1286: return { ...speciesCounts }
   performs a shallow copy with no transformation. If the defensive copy is
   intentional, rename to cloneSpeciesCounts. If it was meant to filter zeros or
   coerce keys, implement that.
5. Repeated error-handling boilerplate Every try/catch in useSimulationSession
   repeats setErrorText(err instanceof Error ? err.message : 'fallback').
   Extract a utility: const captureError = (err: unknown, fallback: string) =>
   setErrorText(err instanceof Error ? err.message : fallback).
6. Magic numbers in rendering code 0.42 (click hit radius), 0.6 (organism draw
   radius), 0.65/0.7 (triangle geometry), 0.14/0.06 (stroke widths) should be
   named constants. Their geometric rationale isn't self-documenting.

Order of implementation:

1. Delete setViewport(next) — clean up React profiling noise
2. useLayoutEffect → useEffect — remove double sync render per tick
3. Dirty-flag the RAF loop — kill idle CPU waste
4. Batch Canvas draw calls — cut per-frame GPU flushes ~10x
5. Precompute hex vertices + facing vectors — already touching draw code from #4
6. Add missing state resets to resetSession — small fix, easier before
   decomposition
7. Decompose useSimulationSession — structural refactor with all behavioral
   changes landed
8. Break up ControlPanel — needs decomposed hooks from #7
9. Incremental computeStableVisibleSpeciesIds — isolated, no dependencies
