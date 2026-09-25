import { gsap } from 'gsap'
import { Flip } from 'gsap/Flip'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useGSAP } from '@gsap/react'

gsap.registerPlugin(useGSAP, Flip, ScrollTrigger)

export const REDUCED = '(prefers-reduced-motion: reduce)'
export const MOTION_OK = '(prefers-reduced-motion: no-preference)'

export { gsap, Flip, ScrollTrigger, useGSAP }
