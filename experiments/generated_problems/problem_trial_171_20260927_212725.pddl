(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    yellow_cube - obj
    blue_can - obj
    sorted_bin - location
    pot - location
  )
  (:init
    (on-table red_can)
    (on-table yellow_cube)
    (on-table blue_can)
  )
  (:goal (and
    (on red_can sorted_bin)
    (on yellow_cube pot)
    (on blue_can sorted_bin)
  ))
)
