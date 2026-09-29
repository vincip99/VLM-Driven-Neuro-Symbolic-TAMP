(define (problem trial-016)
  (:domain manipulation)
  (:objects
    red_can - obj
    blue_can - obj
    green_can - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table red_can)
    (on-table blue_can)
    (on-table green_can)
  )
  (:goal (and
    (on red_can sorting_bin)
    (on blue_can pot)
    (on green_can sorting_bin)
  ))
)
