(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    purple_cube - obj
    blue_can - obj
    pot - location
    sorting_bin - location
  )
  (:init
    (on-table purple_cube)
    (on-table blue_can)
  )
  (:goal (and
    (on purple_cube pot)
    (on blue_can sorting_bin)
  ))
)
