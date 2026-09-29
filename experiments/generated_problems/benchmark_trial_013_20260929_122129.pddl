(define (problem trial-013)
  (:domain manipulation)
  (:objects
    blue_can - obj
    red_can - obj
    yellow_cube - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table blue_can)
    (on-table red_can)
    (on-table yellow_cube)
  )
  (:goal (and
    (on blue_can sorting_bin)
    (on red_can pot)
    (on yellow_cube sorting_bin)
  ))
)
