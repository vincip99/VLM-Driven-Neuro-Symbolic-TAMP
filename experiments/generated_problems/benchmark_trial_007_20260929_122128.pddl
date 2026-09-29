(define (problem trial-007)
  (:domain manipulation)
  (:objects
    red_can - obj
    blue_can - obj
    sorting_bin - location
  )
  (:init
    (on-table red_can)
    (on-table blue_can)
  )
  (:goal (and
    (on red_can sorting_bin)
    (on blue_can sorting_bin)
  ))
)
