(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    yellow_cube - obj
    red_can - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table yellow_cube)
    (on-table red_can)
  )
  (:goal (and
    (on yellow_cube sorting_bin)
    (on red_can pot)
  ))
)
