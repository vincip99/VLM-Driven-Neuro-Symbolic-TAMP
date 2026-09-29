(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_soda_can - obj
    sorting_bin - location
  )
  (:init
    (on-table red_soda_can)
  )
  (:goal (and
    (on red_soda_can sorting_bin)
  ))
)
