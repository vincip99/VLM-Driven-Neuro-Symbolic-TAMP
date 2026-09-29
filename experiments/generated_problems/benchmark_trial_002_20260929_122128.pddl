(define (problem trial-002)
  (:domain manipulation)
  (:objects
    yellow_cube - obj
    pot - location
  )
  (:init
    (on-table yellow_cube)
  )
  (:goal (and
    (on yellow_cube pot)
  ))
)
