(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    yellow_cube - obj
    blue_can - obj
    red_can - obj
    pot - location
    sorting_bin - location
  )
  (:init
    (on-table yellow_cube)
    (on-table blue_can)
    (on-table red_can)
  )
  (:goal (and
    (on yellow_cube pot)
    (on blue_can pot)
    (on red_can sorting_bin)
  ))
)
