(define (problem trial-028)
  (:domain manipulation)
  (:objects
    yellow_cube - obj
    red_can - obj
    blue_can - obj
    pot - location
    sorting_bin - location
  )
  (:init
    (on-table yellow_cube)
    (on-table red_can)
    (on-table blue_can)
  )
  (:goal (and
    (on yellow_cube pot)
    (on red_can sorting_bin)
    (on blue_can pot)
  ))
)
